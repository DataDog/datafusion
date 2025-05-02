use arrow::array::StringBuilder;
use datafusion::arrow::array::{Int32Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use rand::distributions::{Alphanumeric, Uniform};
use rand::{thread_rng, Rng};
use std::sync::Arc;
use arrow::util::pretty::pretty_format_batches;
use datafusion_execution::memory_pool::GreedyMemoryPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;

use datafusion::error::DataFusionError;
use datafusion::functions_aggregate::first_last::last_value_udaf;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::expr::Sort;
use datafusion::logical_expr::simplify::SimplifyInfo;
use datafusion::logical_expr::{expr, function, Accumulator, AggregateUDFImpl};
use datafusion::prelude::Expr;
use datafusion::{
    common::exec_err,
    logical_expr::{function::AccumulatorArgs, Signature, Volatility},
};
use std::any::Any;
use std::fmt::Debug;
use std::ops::Deref;

macro_rules! make_udaf_expr {
    ($EXPR_FN:ident, $($arg:ident)*, $DOC:expr, $AGGREGATE_UDF_FN:ident) => {
        // "fluent expr_fn" style function
        #[doc = $DOC]
        pub fn $EXPR_FN(
            $($arg: datafusion::logical_expr::Expr,)*
        ) -> datafusion::logical_expr::Expr {
            datafusion::logical_expr::Expr::AggregateFunction(datafusion::logical_expr::expr::AggregateFunction::new_udf(
                $AGGREGATE_UDF_FN(),
                vec![$($arg),*],
                false,
                None,
                None,
                None,
            ))
        }
    };
}

macro_rules! make_udaf_expr_and_func {
    ($UDAF:ty, $EXPR_FN:ident, $($arg:ident)*, $DOC:expr, $AGGREGATE_UDF_FN:ident) => {
        make_udaf_expr!($EXPR_FN, $($arg)*, $DOC, $AGGREGATE_UDF_FN);
        create_func!($UDAF, $AGGREGATE_UDF_FN);
    };
    ($UDAF:ty, $EXPR_FN:ident, $DOC:expr, $AGGREGATE_UDF_FN:ident) => {
        // "fluent expr_fn" style function
        #[doc = $DOC]
        pub fn $EXPR_FN(
            args: Vec<datafusion::logical_expr::Expr>,
        ) -> datafusion::logical_expr::Expr {
            datafusion::logical_expr::Expr::AggregateFunction(datafusion::logical_expr::expr::AggregateFunction::new_udf(
                $AGGREGATE_UDF_FN(),
                args,
                false,
                None,
                None,
                None,
            ))
        }

        create_func!($UDAF, $AGGREGATE_UDF_FN);
    };
}

macro_rules! create_func {
    ($UDAF:ty, $AGGREGATE_UDF_FN:ident) => {
        create_func!($UDAF, $AGGREGATE_UDF_FN, <$UDAF>::default());
    };
    ($UDAF:ty, $AGGREGATE_UDF_FN:ident, $CREATE:expr) => {
        paste::paste! {
            /// Singleton instance of [$UDAF], ensures the UDAF is only created once
            /// named STATIC_$(UDAF). For example `STATIC_FirstValue`
            #[allow(non_upper_case_globals)]
            static [< STATIC_ $UDAF >]: std::sync::OnceLock<std::sync::Arc<datafusion::logical_expr::AggregateUDF>> =
                std::sync::OnceLock::new();

            #[doc = concat!("AggregateFunction that returns a [`AggregateUDF`](datafusion_expr::AggregateUDF) for [`", stringify!($UDAF), "`]")]
            pub fn $AGGREGATE_UDF_FN() -> std::sync::Arc<datafusion::logical_expr::AggregateUDF> {
                [< STATIC_ $UDAF >]
                    .get_or_init(|| {
                        std::sync::Arc::new(datafusion::logical_expr::AggregateUDF::from($CREATE))
                    })
                    .clone()
            }
        }
    }
}

make_udaf_expr_and_func!(
    MaxByFunction,
    max_by,
    x y,
    "Returns the value of the first column corresponding to the maximum value in the second column.",
    max_by_udaf
);

pub struct MaxByFunction {
    signature: Signature,
}

impl Debug for MaxByFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("MaxBy")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .field("accumulator", &"<FUNC>")
            .finish()
    }
}
impl Default for MaxByFunction {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxByFunction {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

fn get_min_max_by_result_type(input_types: &[DataType]) -> Result<Vec<DataType>, DataFusionError> {
    match &input_types[0] {
        DataType::Dictionary(_, dict_value_type) => {
            // TODO add checker, if the value type is complex data type
            Ok(vec![dict_value_type.deref().clone()])
        }
        _ => Ok(input_types.to_vec()),
    }
}

impl AggregateUDFImpl for MaxByFunction {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "max_by"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType, DataFusionError> {
        Ok(arg_types[0].to_owned())
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>, DataFusionError> {
        exec_err!("should not reach here")
    }
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>, DataFusionError> {
        get_min_max_by_result_type(arg_types)
    }

    fn simplify(&self) -> Option<function::AggregateFunctionSimplification> {
        let simplify = |mut aggr_func: AggregateFunction, _: &dyn SimplifyInfo| {
            let mut order_by = aggr_func.params.order_by.unwrap_or_default();
            let (second_arg, first_arg) = (aggr_func.params.args.remove(1), aggr_func.params.args.remove(0));

            order_by.push(Sort::new(second_arg, true, false));

            Ok(Expr::AggregateFunction(AggregateFunction::new_udf(
                last_value_udaf(),
                vec![first_arg],
                aggr_func.params.distinct,
                aggr_func.params.filter,
                Some(order_by),
                aggr_func.params.null_treatment,
            )))
        };
        Some(Box::new(simplify))
    }
}

// TODO: use upstream min/max_by
#[tokio::test]
async fn test_highest_score_per_team() -> datafusion::error::Result<()> {
    // Define the schema
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "game_id",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("score", DataType::Int32, true),
        Field::new("team", DataType::Utf8, true),
    ]));

    // Generate 100,000 rows
    let mut rng = thread_rng();
    let num_rows = 100_000;
    let num_teams = 100;
    let str_len = 100;

    let mut game_id_builder = arrow::array::ListBuilder::new(
        StringBuilder::with_capacity(num_rows, num_rows * str_len),
    );
    for _ in 0..num_rows {
        let game_id = (0..str_len)
            .map(|_| rng.sample(&Alphanumeric))
            .map(char::from)
            .collect::<String>();
        game_id_builder.values().append_value(&game_id);
        game_id_builder.append(true);
    }

    let scores: Vec<i32> = (0..num_rows)
        .map(|_| rng.sample(Uniform::new(0, 101)))
        .collect();

    let teams: Vec<String> = (0..num_rows)
        .map(|_| format!("team_{}", rng.sample(Uniform::new(0, num_teams))))
        .collect();

    // Create Arrow arrays
    let game_id_array =
        Arc::new(game_id_builder.finish()) as Arc<dyn arrow::array::Array>;
    let score_array = Arc::new(Int32Array::from(scores)) as Arc<dyn arrow::array::Array>;
    let team_array = Arc::new(StringArray::from(teams)) as Arc<dyn arrow::array::Array>;

    // Create a RecordBatch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![game_id_array, score_array, team_array],
    )?;

    // Create a DataFusion context
    let memory_pool = Arc::new(GreedyMemoryPool::new(100 * num_rows * str_len));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(memory_pool)
        .build_arc()
        .unwrap();
    let ctx = SessionContext::new_with_config_rt(SessionConfig::default(), runtime);

    // Register the RecordBatch as a table
    ctx.register_batch("games", batch)?;
    ctx.register_udaf((*max_by_udaf()).clone());

    // Run the query to get the game with the highest score for each team
    let df = ctx
        .sql(
            "SELECT team, max_by(game_id, score) AS game_id
             FROM games
             GROUP BY team",
        )
        .await?;

    // Run the query
    let record_batches = df.collect().await?;
    println!("{}", pretty_format_batches(&record_batches).unwrap());

    Ok(())
}
