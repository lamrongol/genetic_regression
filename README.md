# genetic_regression
## English
This is nonlinear regression by genetic algorithm, for example:

y = 2.0*x0 + exp(72.0 * x2) + x3^2 + 3.0*√x4 + 1.0/x5 (x1 is not used)

This program chooses which variables are used or not, and function form(e.g. sqrt, log, and so on) by genetic algorithm using BIC(or AIC) as fitness. 

## Japanese
非線形回帰を遺伝的アルゴリズムで行なうプログラムです。例えば以下のような関数があったとします。

y = 2.0*x0 + exp(72.0 * x2) + x3^2 + 3.0*√x4 + 1.0/x5 (x1 is not used)

どの変数を使うか、また関数形(平方根やlogなど)を、BIC(またはAIC)を「適応度」とした遺伝的アルゴリズムによって選択します。

詳しい解説は以下のページを: 遺伝的アルゴリズムによる非線形重回帰分析の変数＆関数選択プログラム http://qiita.com/lamrongol/items/c865b6c10e9b91fbccba
# Example(part of test Code)
```rust
#[test]
fn it_works() {
    const SAMPLE_FILE: &str = "sample_data.tsv";
    let mut reader_builder = csv::ReaderBuilder::new();
    reader_builder.delimiter(b'\t');
 
    let result = fit(
        SAMPLE_FILE,
        reader_builder,
        //express which variables are always plus
        Some(vec![true, true, true, false, true, true]),
        //This means using default genetic algorithm setting
        None,
    );
    if result.is_none(){
        println!("Test Failed");
        return
    }
    let output_file = "result.tsv";
    fs::write(output_file, result.unwrap()).unwrap();
    
    //Use regression result
    let calculator = Calculator::load_file(output_file).unwrap();

    //calculate for new value like following
    let param_vec = &vec![2000.0, 10000.0, 0.06, -18.0, 30000.0, 0.00075];
    let result = calculator.calc(param_vec);
}
```
