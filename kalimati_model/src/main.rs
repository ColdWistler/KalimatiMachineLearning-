
use smartcore::linalg::general::dense_matrix::DenseMatrix;
use smartcore::ensemble::random_forest_regressor::*;
use smartcore::metrics::mean_squared_error;
use std::error::Error;
use csv::Reader;

fn main() -> Result<(), Box<dyn Error>> {
    // Read the CSV file
    let mut rdr = Reader::from_path("data.csv")?;
    let mut features: Vec<f64> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for result in rdr.records().skip(1) {
        let record = result?;
        let quantity: f64 = record[0].parse()?;
        let day: f64 = record[1].parse()?;
        let price: f64 = record[2].parse()?;

        features.push(quantity);
        features.push(day);
        targets.push(price);
    }

    // Reshape to matrix
    let n_samples = targets.len();
    let x = DenseMatrix::from_array(n_samples, 2, &features);
    let y = targets;

    // Train Random Forest Regressor
    let rf = RandomForestRegressor::fit(
        &x,
        &y,
        RandomForestRegressorParameters::default().with_n_trees(100),
    )?;

    // Predict
    let preds = rf.predict(&x)?;

    // Output predictions
    println!("Actual vs Predicted:");
    for (actual, pred) in y.iter().zip(preds.iter()) {
        println!("Actual: {:.2}, Predicted: {:.2}", actual, pred);
    }

    println!("MSE: {:.2}", mean_squared_error(&y, &preds));

    Ok(())
}
