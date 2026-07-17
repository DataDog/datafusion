// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use futures::FutureExt;
use futures::future::BoxFuture;
use std::any::Any;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::future::Future;
use tokio::sync::OnceCell;

/// Type-erased task output used by instrumentation hooks.
pub type ErasedValue = Box<dyn Any + Send>;

/// Type-erased task future used by instrumentation hooks.
pub type ErasedFuture = BoxFuture<'static, ErasedValue>;

/// Type-erased callback that performs the executor spawn.
pub type SpawnCallback<'a> = &'a mut dyn FnMut(ErasedFuture);

/// Runtime selected for a task spawn.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum SpawnTarget<'a> {
    /// Spawn on the ambient Tokio runtime.
    Current,
    /// Spawn on the provided Tokio runtime.
    Runtime(&'a tokio::runtime::Handle),
}

/// A trait for instrumenting task futures, spawn operations, and blocking closures.
pub trait JoinSetTracer: Send + Sync + 'static {
    /// Function pointer type for tracing a future.
    ///
    /// This function takes a boxed future (with its output type erased)
    /// and returns a boxed future (with its output still erased). The
    /// tracer must apply instrumentation without altering the output.
    fn trace_future(&self, fut: ErasedFuture) -> ErasedFuture;

    /// Spawn an erased future through a caller-provided spawn function.
    ///
    /// Implementations may wrap the future or the act of spawning it, but must
    /// invoke `spawn` synchronously exactly once and preserve its return type.
    fn spawn_future(
        &self,
        future: ErasedFuture,
        _target: SpawnTarget<'_>,
        spawn: SpawnCallback<'_>,
    ) {
        spawn(future);
    }

    /// Function pointer type for tracing a blocking closure.
    ///
    /// This function takes a boxed closure (with its return type erased)
    /// and returns a boxed closure (with its return type still erased). The
    /// tracer must apply instrumentation without changing the return value.
    fn trace_block(
        &self,
        f: Box<dyn FnOnce() -> ErasedValue + Send>,
    ) -> Box<dyn FnOnce() -> ErasedValue + Send>;
}

/// A no-op tracer that does not instrument futures, spawns, or closures.
/// This is used as a fallback if no custom tracer is set.
struct NoopTracer;

impl JoinSetTracer for NoopTracer {
    fn trace_future(&self, fut: ErasedFuture) -> ErasedFuture {
        fut
    }

    fn trace_block(
        &self,
        f: Box<dyn FnOnce() -> ErasedValue + Send>,
    ) -> Box<dyn FnOnce() -> ErasedValue + Send> {
        f
    }
}

/// A custom error type for tracer injection failures.
#[derive(Debug)]
pub enum JoinSetTracerError {
    /// The global tracer has already been set.
    AlreadySet,
}

impl Display for JoinSetTracerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            JoinSetTracerError::AlreadySet => {
                write!(f, "The global JoinSetTracer is already set")
            }
        }
    }
}

impl Error for JoinSetTracerError {}

/// Global storage for an injected tracer. If no tracer is injected, a no-op
/// tracer is used instead. Instrumentation hooks therefore remain optional.
static GLOBAL_TRACER: OnceCell<&'static dyn JoinSetTracer> = OnceCell::const_new();

/// A no-op tracer singleton that is returned by [`get_tracer`] if no custom
/// tracer has been registered.
static NOOP_TRACER: NoopTracer = NoopTracer;

/// Return the currently registered tracer, or the no-op tracer if none was
/// registered.
#[inline]
fn get_tracer() -> &'static dyn JoinSetTracer {
    GLOBAL_TRACER.get().copied().unwrap_or(&NOOP_TRACER)
}

/// Set the custom task instrumentor.
///
/// This should be called once at startup. If called more than once, an
/// `Err(JoinSetTracerError)` is returned. If not called at all, a no-op tracer that does nothing
/// is used.
pub fn set_join_set_tracer(
    tracer: &'static dyn JoinSetTracer,
) -> Result<(), JoinSetTracerError> {
    GLOBAL_TRACER
        .set(tracer)
        .map_err(|_set_err| JoinSetTracerError::AlreadySet)
}

/// Optionally instruments a future with custom tracing.
///
/// If a tracer has been injected via `set_tracer`, the future's output is
/// boxed (erasing its type), passed to the tracer, and then downcast back
/// to the expected type. If no tracer is set, the original future is returned.
///
/// # Type Parameters
/// * `T` - The concrete output type of the future.
/// * `F` - The future type.
///
/// # Parameters
/// * `future` - The future to potentially instrument.
pub fn trace_future<T, F>(future: F) -> BoxFuture<'static, T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    // Erase the future’s output type first:
    let erased_future = async move {
        let result = future.await;
        Box::new(result) as ErasedValue
    }
    .boxed();

    // Forward through the global tracer:
    get_tracer()
        .trace_future(erased_future)
        // Downcast from `ErasedValue` back to `T`:
        .map(|any_box| {
            *any_box
                .downcast::<T>()
                .expect("Tracer must preserve the future’s output type!")
        })
        .boxed()
}

/// Instrument and spawn a future through the registered tracer.
///
/// The future first passes through [`trace_future`]. The resulting future and
/// the concrete spawn function then pass through
/// [`JoinSetTracer::spawn_future`], allowing instrumentation that must surround
/// the actual executor spawn.
pub fn spawn_future<T, F, S, R>(future: F, target: SpawnTarget<'_>, spawn: S) -> R
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
    S: FnOnce(BoxFuture<'static, T>) -> R,
{
    spawn_future_with(get_tracer(), future, target, spawn)
}

fn spawn_future_with<T, F, S, R>(
    tracer: &dyn JoinSetTracer,
    future: F,
    target: SpawnTarget<'_>,
    spawn: S,
) -> R
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
    S: FnOnce(BoxFuture<'static, T>) -> R,
{
    let erased_future = async move { Box::new(future.await) as ErasedValue }.boxed();
    let traced_future = tracer.trace_future(erased_future);

    let mut spawn = Some(spawn);
    let mut result = None;
    let mut callback = |future: ErasedFuture| {
        let spawn = spawn
            .take()
            .expect("Task instrumentor invoked the spawn callback more than once");
        let future = future
            .map(|result| {
                *result
                    .downcast::<T>()
                    .expect("Tracer must preserve the future output type")
            })
            .boxed();
        result = Some(spawn(future));
    };

    tracer.spawn_future(traced_future, target, &mut callback);
    result.expect("Task instrumentor did not invoke the spawn callback")
}

/// Optionally instruments a blocking closure with custom tracing.
///
/// If a tracer has been injected via `set_tracer`, the closure is wrapped so that
/// its return value is boxed (erasing its type), passed to the tracer, and then the
/// result is downcast back to the original type. If no tracer is set, the closure is
/// returned unmodified (except for being boxed).
///
/// # Type Parameters
/// * `T` - The concrete return type of the closure.
/// * `F` - The closure type.
///
/// # Parameters
/// * `f` - The blocking closure to potentially instrument.
pub fn trace_block<T, F>(f: F) -> Box<dyn FnOnce() -> T + Send>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // Erase the closure’s return type first:
    let erased_closure = Box::new(|| {
        let result = f();
        Box::new(result) as ErasedValue
    });

    // Forward through the global tracer:
    let traced_closure = get_tracer().trace_block(erased_closure);

    // Downcast from `ErasedValue` back to `T`:
    Box::new(move || {
        let any_box = traced_closure();
        *any_box
            .downcast::<T>()
            .expect("Tracer must preserve the closure’s return type!")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    static SPAWN_HOOK_CALLED: AtomicBool = AtomicBool::new(false);
    static TEST_TRACER: TestTracer = TestTracer;

    struct TestTracer;

    impl JoinSetTracer for TestTracer {
        fn trace_future(&self, future: ErasedFuture) -> ErasedFuture {
            future
        }

        fn spawn_future(
            &self,
            future: ErasedFuture,
            _target: SpawnTarget<'_>,
            spawn: SpawnCallback<'_>,
        ) {
            SPAWN_HOOK_CALLED.store(true, Ordering::Relaxed);
            spawn(future);
        }

        fn trace_block(
            &self,
            f: Box<dyn FnOnce() -> ErasedValue + Send>,
        ) -> Box<dyn FnOnce() -> ErasedValue + Send> {
            f
        }
    }

    struct SpawnCountTracer(usize);

    impl JoinSetTracer for SpawnCountTracer {
        fn trace_future(&self, future: ErasedFuture) -> ErasedFuture {
            future
        }

        fn spawn_future(
            &self,
            future: ErasedFuture,
            _target: SpawnTarget<'_>,
            spawn: SpawnCallback<'_>,
        ) {
            let mut future = Some(future);
            for _ in 0..self.0 {
                let future = future
                    .take()
                    .unwrap_or_else(|| async { Box::new(()) as ErasedValue }.boxed());
                spawn(future);
            }
        }

        fn trace_block(
            &self,
            f: Box<dyn FnOnce() -> ErasedValue + Send>,
        ) -> Box<dyn FnOnce() -> ErasedValue + Send> {
            f
        }
    }

    struct TargetTracer {
        expected: tokio::runtime::Id,
        called: AtomicBool,
    }

    impl JoinSetTracer for TargetTracer {
        fn trace_future(&self, future: ErasedFuture) -> ErasedFuture {
            future
        }

        fn spawn_future(
            &self,
            future: ErasedFuture,
            target: SpawnTarget<'_>,
            spawn: SpawnCallback<'_>,
        ) {
            let SpawnTarget::Runtime(handle) = target else {
                panic!("expected an explicit runtime target");
            };
            assert_eq!(handle.id(), self.expected);
            self.called.store(true, Ordering::Relaxed);
            spawn(future);
        }

        fn trace_block(
            &self,
            f: Box<dyn FnOnce() -> ErasedValue + Send>,
        ) -> Box<dyn FnOnce() -> ErasedValue + Send> {
            f
        }
    }

    #[tokio::test]
    async fn default_spawn_hook_spawns_once() {
        let handle = spawn_future_with(
            &NOOP_TRACER,
            async { 42 },
            SpawnTarget::Current,
            |future| {
                #[expect(clippy::disallowed_methods)]
                let handle = tokio::spawn(future);
                handle
            },
        );
        assert_eq!(handle.await.unwrap(), 42);
    }

    #[test]
    fn spawn_future_forwards_explicit_target() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tracer = TargetTracer {
            expected: runtime.handle().id(),
            called: AtomicBool::new(false),
        };
        let handle = spawn_future_with(
            &tracer,
            async { 42 },
            SpawnTarget::Runtime(runtime.handle()),
            |future| runtime.spawn(future),
        );

        assert_eq!(runtime.block_on(handle).unwrap(), 42);
        assert!(tracer.called.load(Ordering::Relaxed));
    }

    #[tokio::test]
    #[should_panic(expected = "did not invoke the spawn callback")]
    async fn spawn_future_rejects_zero_spawns() {
        let tracer = SpawnCountTracer(0);
        drop(spawn_future_with(
            &tracer,
            async { 42 },
            SpawnTarget::Current,
            |future| {
                #[expect(clippy::disallowed_methods)]
                let handle = tokio::spawn(future);
                handle
            },
        ));
    }

    #[tokio::test]
    #[should_panic(expected = "more than once")]
    async fn spawn_future_rejects_multiple_spawns() {
        let tracer = SpawnCountTracer(2);
        drop(spawn_future_with(
            &tracer,
            async { 42 },
            SpawnTarget::Current,
            |future| {
                #[expect(clippy::disallowed_methods)]
                let handle = tokio::spawn(future);
                handle
            },
        ));
    }

    #[tokio::test]
    async fn spawn_future_uses_registered_hook() {
        set_join_set_tracer(&TEST_TRACER).unwrap();
        let handle = spawn_future(async { 42 }, SpawnTarget::Current, |future| {
            #[expect(clippy::disallowed_methods)]
            let handle = tokio::spawn(future);
            handle
        });
        assert_eq!(handle.await.unwrap(), 42);
        assert!(SPAWN_HOOK_CALLED.load(Ordering::Relaxed));
    }
}
