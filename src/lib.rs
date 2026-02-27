mod gene;
mod individual;

use crate::gene::Gene;
use crate::individual::Individual;
use ndarray::Array1;
use rand::prelude::SliceRandom;
use rand::{RngExt, rng};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{
    LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
};
use std::cmp::min;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::slice::Iter;

pub enum Evaluation {
    StepwiseAic,
    StepwiseBic,
    // R2,
}

pub struct AlgorithmSetting {
    pub evaluation: Evaluation,
    pub individual_num: usize,
    pub top_selection_num: usize,
    pub mutation_rate: f64,
    pub min_loop_cnt: usize,
    pub max_loop_cnt: usize,
    pub stop_diff_rate: f64,
}

impl AlgorithmSetting {
    pub fn default() -> Self {
        AlgorithmSetting {
            evaluation: Evaluation::StepwiseBic,
            individual_num: 500,
            top_selection_num: 5,
            mutation_rate: 0.30,
            min_loop_cnt: 3,
            max_loop_cnt: 30,
            stop_diff_rate: 0.000001,
        }
    }
}
pub struct Calculator {
    individual: Individual,
}

impl Calculator {
    pub fn load_file(tsv_file: &str) -> Result<Calculator, &str> {
        let individual = Individual::load(&tsv_file);
        if individual.is_err() {
            return Err(individual.err().unwrap());
        }
        Ok(Calculator {
            individual: individual?,
        })
    }
    pub fn save_file(output_file: &str, fitting_result: String) {
        fs::write(output_file, fitting_result).unwrap();
    }

    pub fn calc(&self, params: &Vec<f64>) -> f64 {
        self.individual.calc(params)
    }
}

/// if `non_negative_vec` is `None`, only minus acceptable function form is used(such as x^2), log and sqrt are not used.
/// For `Some(non_negative_vec)`, length can be 1, if all parameters' `non_negative` are same(e.g. `[true]` is automatically converted to `&vec![true; param_num]`)
///` genetic_setting` can be `None` if you are not interested in genetic algorithm
pub fn fit(
    dataset: &Dataset,
    original_data_info: &OriginalDataInfo,
    genetic_setting: Option<AlgorithmSetting>,
) -> Result<String, String> {
    let setting = genetic_setting.unwrap_or_else(|| AlgorithmSetting::default());

    let non_negative_list = original_data_info
        .min_list
        .iter()
        .map(|min| *min >= 0.0)
        .collect::<Vec<_>>();

    let scale_list = original_data_info
        .median_list
        .iter()
        .map(|median| if *median == 0.0 { 1.0 } else { 1.0 / median })
        .collect::<Vec<_>>();

    let mut individuals = vec![];
    for _i in 0..setting.individual_num {
        let individual = Individual::new(&scale_list, &non_negative_list);
        individuals.push(individual);
    }
    individuals.par_iter_mut().for_each(|mut individual| {
        let result = calc_evaluation(&mut individual, &dataset, &original_data_info);
        if result.is_ok() {
            *individual = result.unwrap()
        }
    });
    individuals.retain(|i| i.is_fitted);
    if individuals.len() < setting.individual_num {
        dbg!(individuals.len());
    }

    let mut pre_best_evaluation = match setting.evaluation {
        Evaluation::StepwiseAic => {
            individuals
                .sort_unstable_by(|a, b| a.aic.unwrap().partial_cmp(&b.aic.unwrap()).unwrap());
            individuals[0].aic
        }
        Evaluation::StepwiseBic => {
            individuals
                .sort_unstable_by(|a, b| a.bic.unwrap().partial_cmp(&b.bic.unwrap()).unwrap());
            individuals[0].bic
        }
    }
    .unwrap();

    let mut generation_idx = 0;
    let best: Individual;
    loop {
        generation_idx += 1;
        dbg!(generation_idx);

        let mut next_individuals: Vec<Individual> = vec![];
        //Select top and mating
        // next_individuals.append(&mut individuals[0..setting.top_selection_num].to_vec());
        for i in 0..setting.top_selection_num {
            next_individuals.push(individuals[i].clone());
        }

        for i in setting.top_selection_num..individuals.len() {
            if rand::random_range(0.0..1.0) < (setting.top_selection_num as f64) / (i as f64) {
                next_individuals.push(individuals[i].clone())
            }
        }

        let survivor_count = next_individuals.len();
        while next_individuals.len() < setting.individual_num {
            let mother_idx = rng().random_range(0..survivor_count);
            let mut father_idx = rng().random_range(0..survivor_count);
            if father_idx == mother_idx {
                father_idx = (father_idx + 1) % survivor_count
            }
            let (child1, child2) = next_individuals[mother_idx].cross(
                next_individuals[father_idx].clone(),
                &scale_list,
                &non_negative_list,
            );
            next_individuals.push(child1);
            if next_individuals.len() < setting.individual_num {
                next_individuals.push(child2);
            }
        }
        while next_individuals.len() < setting.individual_num {
            next_individuals.push(Individual::new(&scale_list, &non_negative_list));
        }

        //mutation
        next_individuals[setting.top_selection_num..]
            .par_iter_mut()
            .for_each(|i| {
                if rand::random_range(0.0..1.0) < setting.mutation_rate {
                    let mutation_idx = rand::random_range(0..original_data_info.param_cnt());
                    let gene = Gene::get_random_gene(
                        scale_list[mutation_idx],
                        non_negative_list[mutation_idx],
                    );
                    i.set_gene(mutation_idx, gene);
                }
            });

        individuals = next_individuals;
        individuals.par_iter_mut().for_each(|mut individual| {
            let result = calc_evaluation(&mut individual, &dataset, &original_data_info);
            if result.is_ok() {
                *individual = result.unwrap();
            }
        });
        individuals.retain(|i| i.is_fitted);
        if individuals.len() < setting.individual_num {
            dbg!(individuals.len());
        }

        let best_eval = match setting.evaluation {
            Evaluation::StepwiseAic => {
                individuals.sort_by(|a, b| a.aic.unwrap().partial_cmp(&b.aic.unwrap()).unwrap());
                individuals[0].aic
            }
            Evaluation::StepwiseBic => {
                individuals.sort_by(|a, b| a.bic.unwrap().partial_cmp(&b.bic.unwrap()).unwrap());
                individuals[0].bic
            }
        }
        .unwrap();

        if best_eval == 0.0 || pre_best_evaluation == 0.0 {
            if best_eval == 0.0 || pre_best_evaluation == 0.0 {
                // println!("Break because evaluation metrics doesn't change");
                break;
            } else {
                println!(
                    "Continue loop because evaluation metrics is zero: {} {}",
                    best_eval, pre_best_evaluation
                );
            }
        } else if best_eval < 0.0 && pre_best_evaluation > 0.0 {
            println!(
                "Continue loop because evaluation metrics sign is different: {} {}",
                best_eval, pre_best_evaluation
            );
        } else {
            let improvement_rate = if best_eval > 0.0 && pre_best_evaluation > 0.0 {
                1.0 - best_eval / pre_best_evaluation
            } else {
                1.0 - pre_best_evaluation / best_eval
            };
            dbg!(improvement_rate);
            // if loop_count > MIN_LOOP_COUNT && diff_rate < STOP_DIFF_RATE {
            if improvement_rate < setting.stop_diff_rate {
                break;
            }
        }
        if generation_idx < setting.min_loop_cnt {
            continue;
        } else if generation_idx > setting.max_loop_cnt {
            break;
        }
        pre_best_evaluation = best_eval;
    }
    best = individuals[0].clone();

    println!("Finished!");
    Ok(best.format(
        &original_data_info.param_names,
        &original_data_info.median_list,
    ))
}

fn calc_evaluation(
    individual: &mut Individual,
    dataset: &Dataset,
    original_data_info: &OriginalDataInfo,
) -> Result<Individual, String> {
    let param_num = original_data_info.param_cnt();

    let mut idx_rel_list = vec![];
    let mut dim_sum = 0;
    for (idx, gene) in individual.gene_list().iter().enumerate() {
        if *gene != Gene::Unused {
            idx_rel_list.push(idx);
            dim_sum += gene.dim();
        }
    }

    let mut x_matrix: Vec<Vec<f64>> = Vec::with_capacity(dataset.data_cnt());
    let mut target_list: Vec<f64> = Vec::with_capacity(dataset.data_cnt());
    for data_item in dataset.iter() {
        target_list.push(data_item.target);
        let mut vec = vec![];
        for (idx, gene) in individual.gene_list().iter().enumerate() {
            if *gene != Gene::Unused {
                let val = gene.calc(data_item.params[idx]).unwrap();
                if !is_usual(val) {
                    // println!("{}, {}",idx, gene.name());
                    return Err(String::from("includes unusual number(NaN or Infinity)"));
                }
                vec.push(val);
            }
        }
        x_matrix.push(vec);
    }

    //All gene is `Unused`
    if idx_rel_list.len() == 0 {
        individual.aic = Some(f64::MAX);
        individual.bic = Some(f64::MAX);
        return Ok(individual.clone());
    }

    let matrix = DenseMatrix::from_2d_vec(&x_matrix).unwrap();
    let lr_result = LinearRegression::fit(
        &matrix,
        &target_list,
        LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
    );

    let lr: LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>> = match lr_result {
        Ok(lr) => lr,
        Err(failed) => {
            return Err(failed.to_string());
        }
    };
    let y_hat = lr.predict(&matrix).unwrap();

    //residual sum of squares
    let rss = target_list
        .iter()
        .zip(y_hat.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>();
    let data_num = target_list.len() as f64;

    let dim_sum = dim_sum as f64;
    let d = if rss == 0.0 {
        0.0
    } else {
        data_num * (rss / data_num).ln()
    };
    //@see https://qiita.com/WolfMoon/items/6164c09b93ca043690b3
    individual.aic = Some(d + 2.0 * (dim_sum + 1.0));
    individual.bic = Some(d + data_num.ln() * (dim_sum + 1.0));
    individual.intercept = Some(*lr.intercept());
    let mut coe_list: Vec<Option<f64>> = vec![None; param_num];
    for (idx, coe) in lr.coefficients().iter().enumerate() {
        coe_list[idx_rel_list[idx]] = Some(*coe);
    }
    individual.coe_list = coe_list;
    individual.is_fitted = true;
    Ok(individual.to_owned())
}

fn is_usual(x: f64) -> bool {
    !(x.is_nan() || x.is_infinite() || x.is_subnormal())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    pub target: f64,
    pub params: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    vec: Vec<DataItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalDataInfo {
    pub min_list: Vec<f64>,
    pub max_list: Vec<f64>,
    pub median_list: Vec<f64>,
    pub target_min: f64,
    pub target_max: f64,

    pub param_names: Vec<String>,
}

impl OriginalDataInfo {
    pub fn param_cnt(&self) -> usize {
        self.param_names.len()
    }
    pub fn restore_target(&self, normalized_target: f64) -> f64 {
        normalized_target * (self.target_max - self.target_min) + self.target_min
    }
}

impl Dataset {
    pub fn data_cnt(&self) -> usize {
        self.vec.len()
    }

    pub fn split(&mut self, train_rate: f64, test_size: usize) -> (Self, Self, Self) {
        let data_cnt = self.data_cnt();
        let (test, left) = self
            .vec
            .partial_shuffle(&mut rand::rng(), min(test_size, data_cnt / 100));
        let (train, validation) = left.split_at(((left.len() as f64) * train_rate) as usize);
        (
            Dataset {
                vec: train.to_vec(),
            },
            Dataset {
                vec: validation.to_vec(),
            },
            Dataset { vec: test.to_vec() },
        )
    }
    pub fn iter(&self) -> Iter<'_, DataItem> {
        self.vec.iter()
    }

    pub fn to_ndarray(&self, data_info: &OriginalDataInfo) -> (ndarray::Array2<f64>, Array1<f64>) {
        let mut x = vec![];
        let mut y = vec![];
        for item in self.vec.iter() {
            y.push(item.target);
            x.append(&mut item.params.to_owned());
        }
        let x =
            ndarray::Array2::from_shape_vec((self.data_cnt(), data_info.param_cnt()), x).unwrap();
        (x, Array1::from(y))
    }
}

pub struct DataLoadSetting {
    pub max_data_num: usize,
    pub normalize: bool,
    pub ignore_variables: HashSet<String>,
}

impl DataLoadSetting {
    pub fn default() -> Self {
        DataLoadSetting {
            max_data_num: 500_000,
            normalize: false,
            ignore_variables: HashSet::new(),
        }
    }
}

pub fn load_dataset(
    input_file: &str,
    csv_reader_builder: csv::ReaderBuilder,
    setting: Option<DataLoadSetting>,
) -> Result<(Dataset, OriginalDataInfo), String> {
    let setting = setting.unwrap_or_else(|| DataLoadSetting::default());

    let mut csv_reader = csv_reader_builder.from_path(input_file).unwrap();
    let mut ignore_column_idxes = HashSet::new();
    let mut target_idx = 0;
    if !csv_reader.has_headers() {
        return Err("csv file does not contain headers".to_string());
    }

    let headers = csv_reader.headers().unwrap();
    let mut param_names = vec![];
    let mut is_first = true;
    for (i, name) in headers.iter().enumerate() {
        if setting.ignore_variables.contains(name) {
            ignore_column_idxes.insert(i);
            continue;
        }
        if is_first {
            target_idx = i;
            is_first = false;
            continue;
        }
        param_names.push(name.to_string());
    }
    let param_cnt = param_names.len();
    let mut min_list = vec![f64::INFINITY; param_cnt];
    let mut max_list = vec![f64::NEG_INFINITY; param_cnt];
    let all_data_count = count_data_num(input_file);
    if all_data_count < 1000 {
        return Err(format!(
            "data count is too small: {}, no result",
            all_data_count
        ));
    }
    let data_cnt = if all_data_count > setting.max_data_num {
        setting.max_data_num
    } else {
        all_data_count
    };
    if data_cnt < 1000 {
        eprintln!("data count is too small: {}, no result", data_cnt);
    }

    let mut target_min = f64::INFINITY;
    let mut target_max = f64::NEG_INFINITY;
    let mut target_list = Vec::with_capacity(data_cnt);
    let mut median_list = Vec::with_capacity(data_cnt);
    let mut vertical_parameter_list = vec![Vec::with_capacity(data_cnt); param_cnt];

    let mut csv_reader = csv_reader_builder.from_path(input_file).unwrap();
    for result in csv_reader.records().skip(all_data_count - data_cnt) {
        let record = result.unwrap();
        let target = record.get(target_idx).unwrap().parse::<f64>().unwrap();
        if target < target_min {
            target_min = target;
        }
        if target > target_max {
            target_max = target;
        }
        target_list.push(target);

        let mut idx = 0;
        for (i, rec) in record.iter().enumerate() {
            if i == target_idx || ignore_column_idxes.contains(&i) {
                continue;
            }
            let param = rec.parse::<f64>().unwrap();
            if param < min_list[idx] {
                min_list[idx] = param;
            }
            if param > max_list[idx] {
                max_list[idx] = param;
            }
            vertical_parameter_list[idx].push(param);

            idx += 1;
        }
    }

    let data_num = if data_cnt > setting.max_data_num {
        setting.max_data_num
    } else {
        data_cnt
    };

    for mut v in vertical_parameter_list.to_owned() {
        let (_, median, _) =
            v.select_nth_unstable_by(data_num / 2, |a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        median_list.push(median.to_owned());
    }

    if setting.normalize {
        target_list
            .iter_mut()
            .for_each(|x| *x = (*x - target_min) / (target_max - target_min));
    }

    let mut vec = Vec::with_capacity(data_cnt);
    for i in 0..data_cnt {
        let mut params = Vec::with_capacity(param_cnt);
        for j in 0..param_cnt {
            let val = vertical_parameter_list[j][i];
            if setting.normalize {
                params.push((val - min_list[j]) / (max_list[j] - min_list[j]));
            } else {
                params.push(val);
            }
        }
        vec.push(DataItem {
            target: target_list[i],
            params,
        });
    }

    Ok((
        Dataset { vec },
        OriginalDataInfo {
            min_list,
            max_list,
            median_list,
            target_min,
            target_max,
            param_names,
        },
    ))
}

fn count_data_num(path: &str) -> usize {
    let file = File::open(path).unwrap();
    let br = BufReader::new(file);
    br.lines().count() - 1 //minus header line
}

#[cfg(test)]
mod tests {
    use crate::{Calculator, fit, load_dataset};
    use rand::{RngExt, rng};

    #[test]
    fn it_works() {
        const SAMPLE_FILE: &str = "sample_data.tsv";
        const DATA_NUM: i32 = 2000;

        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(SAMPLE_FILE)
            .unwrap();
        writer
            .write_record(["y", "x0", "x1", "x2", "x3", "x4", "x5"])
            .unwrap();

        for _i in 0..DATA_NUM {
            let x0 = rng().random_range(1000.0..10000.0);
            let x1 = rng().random_range(5000.0..50000.0);
            let x2: f64 = rng().random_range(0.01..0.15);
            let x3: f64 = rng().random_range(-62.0..62.0);
            let x4: f64 = rng().random_range(1000.0..80001000.0);
            let x5 = rng().random_range(0.0..50.0);
            let noise = rng().random_range(-0.50..0.50) * 1000.0;

            let y = function(x0, x1, x2, x3, x4, x5) + noise;
            writer
                .write_record([
                    y.to_string(),
                    x0.to_string(),
                    x1.to_string(),
                    x2.to_string(),
                    x3.to_string(),
                    x4.to_string(),
                    x5.to_string(),
                ])
                .unwrap();
        }
        writer.flush().unwrap();

        let mut reader_builder = csv::ReaderBuilder::new();
        reader_builder.delimiter(b'\t');
        let (dataset, data_info) = load_dataset(SAMPLE_FILE, reader_builder, None).unwrap();

        let result = fit(&dataset, &data_info, None);

        let output_file = "result.tsv";
        // let output_path = Path::new(output_file);
        // match output_path.parent() {
        //     Some(dir) => fs::create_dir_all(dir).unwrap(),
        //     None => {}
        // }
        if result.is_err() {
            println!("Test Failed");
            return;
        }
        Calculator::save_file(output_file, result.unwrap());

        //calculate for new data
        let calculator = Calculator::load_file(output_file).unwrap();
        let result = calculator.calc(&vec![2000.0, 10000.0, 0.06, -18.0, 30000.0, 0.00075]);
        let expected = function(2000.0, 10000.0, 0.06, -18.0, 30000.0, 0.00075);

        assert!(dbg!((result - expected) / expected).abs() < 0.05);
    }

    fn function(x0: f64, _x1: f64, x2: f64, x3: f64, x4: f64, x5: f64) -> f64 {
        2.0 * x0 + (72.0 * x2).exp() + x3.powi(2) + 3.0 * x4.sqrt() + 5000.0 * (-0.5 * x5).exp()
    }
}
