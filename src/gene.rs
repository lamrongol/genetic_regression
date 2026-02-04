use std::str::FromStr;
use rand::{Rng, rng};
use strum::EnumCount;
use strum_macros::{EnumCount, EnumString, FromRepr, IntoStaticStr};

//Developers must think to change this when changing Gene enum.
const ACCEPT_MINUS_GENE_CNT: usize = 7;

#[derive(FromRepr, Debug, PartialEq, EnumCount, EnumString, IntoStaticStr)]
pub(crate) enum Gene {
    Unused,
    Linear,
    Squared,
    Cubed,
    //These may result INFINITY
    Exp(f64),
    ExpMinus(f64),
    Inverse,
    //Following accept only plus, above genes also accept minus and count is `ACCEPT_MINUS_GENE_CNT`
    Inverse1plus(f64),
    Sqrt,
    Log,//This may result INFINITY
    Log1plus(f64),
}
impl Gene {
    pub(crate) fn name(&self) -> String {
        let name:&str = self.into();
        name.to_string()
    }
    pub(crate) fn dim(&self) -> usize {
        match self {
            Gene::Unused => 0,
            Gene::Linear => 1,
            Gene::Squared => 1,
            Gene::Cubed => 1,
            Gene::Exp(_s) => 2,
            Gene::ExpMinus(_s) => 2,
            Gene::Inverse => 1,
            Gene::Inverse1plus(_s) => 2,
            Gene::Sqrt => 1,
            Gene::Log => 1,
            Gene::Log1plus(_s) => 2,
        }
    }
    #[allow(dead_code)]
    pub(crate) fn accept_only_plus(&self) -> bool {
        match self {
            Gene::Unused | Gene::Linear | Gene::Squared | Gene::Cubed | Gene::Inverse  => false,
                Gene::Exp(_s) | Gene::ExpMinus(_s) => false,
            Gene::Sqrt | Gene::Log  => true,
             Gene::Log1plus(_s) | Gene::Inverse1plus(_s) => true,
        }
    }

    pub(crate) fn calc(&self, x: f64) -> Option<f64> {
        match self {
            Gene::Unused => None,
            Gene::Linear => Some(x),
            Gene::Squared => Some(x.powi(2)),
            Gene::Cubed => Some(x.powi(3)),
            Gene::Exp(s) => Some((s * x).exp()),
            Gene::ExpMinus(s) => Some((-s * x).exp()),
            Gene::Inverse => Some(1.0 / x),
            Gene::Inverse1plus(s) => Some(1.0 / (1.0+s*x)),
            Gene::Sqrt => Some(x.sqrt()),
            Gene::Log => Some(x.ln()),
            Gene::Log1plus(s) => Some((1.0 + s * x).ln()),
        }
    }
    //Avoid Infinity
    // pub(crate) fn calc_suppressing_outlier(&self, x: f64) -> f64 {
    //     let val = match self {
    //         Gene::Unused => 0.0,
    //         Gene::Linear => x,
    //         Gene::Squared => x.powi(2),
    //         Gene::Cubed => x.powi(3),
    //         Gene::Exp(s) => (s * x).exp(),
    //         Gene::ExpMinus(s) => (-s * x).exp(),
    //         Gene::Inverse => 1.0 / x,
    //         Gene::Inverse1plus(s) => 1.0 / (1.0+s*x),
    //         Gene::Sqrt => x.sqrt(),
    //         Gene::Log => x.ln(),
    //         Gene::Log1plus(s) => (1.0 + s * x).ln(),
    //     };
    //     if val.is_infinite() {
    //         f32::MAX as f64
    //     }else{
    //         val
    //     }
    // }

    pub(crate) fn get_scale_factor(&self) -> Option<f64> {
        if self.dim() < 2 {
            None
        } else {
            match self {
                Gene::Exp(scale_factor)
                | Gene::ExpMinus(scale_factor)
                | Gene::Log1plus(scale_factor)
                | Gene::Inverse1plus(scale_factor)=> Some(*scale_factor),
                _ => {
                    panic!("Code for {} is not written", self.name());
                }
            }
        }
    }

    pub(crate) fn to_string(&self) -> String {
        match self.dim() {
            0 => self.name(),
            1 =>  format!("{}\t",self.name()),
            2 => match self {
                Gene::Exp(scale_factor)
                | Gene::ExpMinus(scale_factor)
                | Gene::Log1plus(scale_factor)
                | Gene::Inverse1plus(scale_factor)=> {
                    format!("{}\t{}", self.name(), scale_factor)
                }
                _ => {
                    panic!("Code for {} is not written", self.name());
                }
            },
            _ => panic!("Code for {} is not written", self.name()),
        }
    }

    pub(crate) fn get_random_gene(mut scale_factor: f64, is_plus: bool) -> Self {
        let minus_idx = if is_plus { Gene::COUNT } else { ACCEPT_MINUS_GENE_CNT };
        let gene = Gene::from_repr(rng().random_range(0..minus_idx)).unwrap();
        if gene.dim() == 2 {
            if rand::rng().random_bool(0.5) {
                scale_factor *= rand::rng().random_range(0.1..1.0);
            } else {
                scale_factor *= rand::rng().random_range(1.0..10.0);
            }

            match gene {
                Gene::Exp(_s) => Gene::Exp(scale_factor),
                Gene::ExpMinus(_s) => Gene::ExpMinus(scale_factor),
                Gene::Log1plus(_s) => Gene::Log1plus(scale_factor),
                Gene::Inverse1plus(_s)=> Gene::Inverse1plus(scale_factor),
                _ => {
                    panic!("Code for {} is not written", gene.name());
                }
            }
        } else {
            gene
        }
    }
    pub(crate) fn cross_different_scale(&self, partner: Gene) -> (Self, Self) {
        let s1 = self.get_scale_factor().unwrap();
        let s2 = partner.get_scale_factor().unwrap();
        let average = (s1 + s2) / 2.0;
        let another = if rand::rng().random_bool(0.5) {
            s1.min(s2)*0.95
        } else {
            s1.max(s2)*1.05
        };

        match self {
            Gene::Exp(_s) => (Gene::Exp(average), Gene::Exp(another)),
            Gene::ExpMinus(_s) => (Gene::ExpMinus(average), Gene::ExpMinus(another)),
            Gene::Log1plus(_s) => (Gene::Log1plus(average), Gene::Log1plus(another)),
            Gene::Inverse1plus(_s) => (Gene::Inverse1plus(average), Gene::Inverse1plus(another)),
            _ => {
                panic!("Code for {} is not written", self.name())
            },
        }
    }

    pub(crate) fn load_from_str(string: &str, scale: Option<f64>) -> Gene {
        match string {
            "Exp" => Gene::Exp(scale.unwrap()),
            "ExpMinus" => Gene::ExpMinus(scale.unwrap()),
            "Log1plus" => Gene::Log1plus(scale.unwrap()),
            "Inverse1plus" => Gene::Inverse1plus(scale.unwrap()),
            _ => Gene::from_str(string).unwrap_or_else(|_| panic!("Code for {} is not written", string)),
        }
    }
}

impl Clone for Gene {
    fn clone(&self) -> Self {
        match self {
            Gene::Unused => Gene::Unused,
            Gene::Linear => Gene::Linear,
            Gene::Squared => Gene::Squared,
            Gene::Cubed => Gene::Cubed,
            Gene::Exp(s) => Gene::Exp(*s),
            Gene::ExpMinus(s) => Gene::ExpMinus(*s),
            Gene::Inverse => Gene::Inverse,
            Gene::Inverse1plus(s) => Gene::Inverse1plus(*s),
            Gene::Sqrt => Gene::Sqrt,
            Gene::Log => Gene::Log,
            Gene::Log1plus(s) => Gene::Log1plus(*s),
        }
    }
}
