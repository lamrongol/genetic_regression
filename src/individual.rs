use rand::{Rng, rng};
use std::fmt::Display;
use std::slice::Iter;
use crate::gene::Gene;

pub(crate) struct Individual {
    gene_num: usize,
    gene_list: Vec<Gene>,
    // minus_possible: Vec<bool>,
    pub(crate) coe_list: Vec<Option<f64>>,
    pub(crate) intercept: Option<f64>,

    pub(crate) aic: Option<f64>,
    pub(crate) bic: Option<f64>,
    // pub(crate) r2: Option<f64>,

    pub(crate) is_fitted: bool,
}

impl Individual {
    pub(crate) fn cross(
        &self,
        partner: Individual,
        scale_list: &Vec<f64>,
        is_plus_list: &Vec<bool>,
    ) -> (Self, Self) {
        let gene_num = self.gene_num;
        let start_idx = rng().random_range(0..gene_num);
        let mut end_idx = rng().random_range(0..gene_num);
        if end_idx == start_idx {
            end_idx = (start_idx + 1) % gene_num
        }
        let mut child1 = self.clone();
        let mut child2 = partner.clone();

        let mut changed = false;
        let mut i = start_idx;
        while i != end_idx {
            if child1.gene_list[i].name() != child2.gene_list[i].name() {
                changed = true;
                child1.gene_list[i] = partner.gene_list[i].clone(); //if (isPlus == null || isPlus(i) || GeneManager.allowMinus(father.genes(i)))
                child2.gene_list[i] = self.gene_list[i].clone() //if (isPlus == null || isPlus(i) || GeneManager.allowMinus(this.genes(i)))
            } else {
                let gene = &child1.gene_list[i];
                if gene.dim() == 2 {
                    changed = true;
                    let (gene1, gene2) = gene.cross_different_scale(child2.gene_list[i].clone());

                    child1.gene_list[i] = gene1;
                    child2.gene_list[i] = gene2;
                }
            }
            i = (i + 1) % gene_num
        }
        if !changed {
            //Mutation
            child1.gene_list[i] = Gene::get_random_gene(scale_list[i], is_plus_list[i]);
            child2.gene_list[i] = Gene::get_random_gene(scale_list[i], is_plus_list[i]);
        }
        (child1, child2)
    }
    pub(crate) fn calc(&self, params: &Vec<f64>) -> f64 {
        let mut v = self.intercept.unwrap();
        for (i, param) in params.iter().enumerate() {
            let gene = &self.gene_list[i];
            v += if *gene != Gene::Unused {
                gene.calc(*param).unwrap()
            } else {
                0.0
            };
        }
        v
    }
    pub(crate) fn gene_iter(&self) -> Iter<'_, Gene> {
        self.gene_list.iter()
    }
    pub(crate) fn set_gene(&mut self, idx: usize, gene: Gene) {
        self.gene_list[idx] = gene;
    }
    pub(crate) fn new(scale_list: &Vec<f64>, is_plus_list: &Vec<bool>) -> Self {
        let gene_num = scale_list.len();
        let mut gene_list = vec![];
        for i in 0..scale_list.len() {
            gene_list.push(Gene::get_random_gene(scale_list[i], is_plus_list[i]));
        }
        Individual {
            gene_num,
            gene_list,
            coe_list: vec![None; gene_num],
            intercept: None,
            aic: None,
            bic: None,
            is_fitted: false,
        }
    }
    pub(crate) fn format(&self, param_names: Option<Vec<&str>>) -> String {
        if !self.is_fitted {
            let mut s = String::new();
            for i in 0..self.gene_num {
                let gene = &self.gene_list[i];
                s.push_str(&format!("{}\n", gene.to_string()));
            }
            s
        } else {
            let mut s = String::from("#name\tCoefficient\tFunction\tScaling Factor(if exists)\n");
            s.push_str(&format!("[Intercept]\t{}\n", self.intercept.unwrap()));
            let coe_list = &self.coe_list;

            for i in 0..self.gene_num {
                let gene = &self.gene_list[i];
                let line = if gene.dim() == 0 {
                   format!("\t{}\n", gene.to_string())
                } else {
                    format!(
                        "{}\t{}\n",
                        coe_list[i].unwrap(),
                        gene.to_string()
                    )
                };
                match param_names {
                    Some(ref param_names) => s.push_str(&format!("{}\t{}", param_names[i], line)),
                    None => {s.push_str(line.as_str())}
                }
            }
            if self.aic.is_some() {
                s.push_str(&format!("#AIC={}\n", self.aic.unwrap()));
            }
            if self.bic.is_some() {
                s.push_str(&format!("#BIC={}\n", self.bic.unwrap()));
            }
            s
        }
    }
    pub(crate) fn load(tsv_file: &str) -> Self {
        let mut tsv_reader = csv::ReaderBuilder::new()
            .flexible(true)
            .delimiter(b'\t')
            .from_path(tsv_file)
            .unwrap();
        let first = tsv_reader.records().nth(0).unwrap().unwrap();
        let intercept = Some(first[1].parse::<f64>().unwrap());
        let mut gene_list = vec![];
        let mut coe_list = vec![];
        for record in tsv_reader.records() {
            let line = record.unwrap();
            if line[0].starts_with("#") {
                continue;
            }

            let coe = match line[1].parse::<f64>() {
                Ok(c) => Some(c),
                Err(_) => None,
            };
            coe_list.push(coe);
            let scale: Option<f64> = if line.len() == 4 && !line[3].is_empty() {
                Some(line[3].parse::<f64>().unwrap())
            } else {
                None
            };
            let gene = Gene::load_from_str(&line[2], scale);
            gene_list.push(gene);
        }
        Individual {
            gene_num: gene_list.len(),
            gene_list,
            coe_list,
            intercept,
            aic: None,
            bic: None,
            is_fitted: true
        }
    }
}
impl Clone for Individual {
    fn clone(&self) -> Self {
        Individual {
            gene_num: self.gene_num,
            gene_list: self.gene_list.clone(),
            coe_list: self.coe_list.clone(),
            intercept: self.intercept,
            aic: self.aic,
            bic: self.bic,
            is_fitted: self.is_fitted,
        }
    }
}
impl Display for Individual {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format(None))
    }
}
