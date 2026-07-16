use std::path::PathBuf;

use clap::Parser;
use rust_preprocessor::preprocess_dataset_impl;

#[derive(Parser, Debug)]
#[command(about = "将胸部 X 光 JPEG 图像预处理为 .npy 缓存 (uint8, 256x256)")]
struct Args {
    #[arg(
        long,
        env = "DATASET_ROOT",
        default_value = "/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray"
    )]
    root: PathBuf,

    #[arg(long, env = "CACHE_DIR", default_value = "cache")]
    out: PathBuf,

    #[arg(long, default_value_t = 256)]
    size: u32,

    #[arg(
        long,
        default_value_t = true,
        help = "将 kaggle val 目录并入 train（kaggle 兼容）"
    )]
    merge_val: bool,
}

fn main() {
    let args = Args::parse();
    match preprocess_dataset_impl(
        args.root.to_str().unwrap(),
        args.out.to_str().unwrap(),
        args.size,
        args.merge_val,
    ) {
        Ok((train_count, test_count)) => {
            println!("处理完成: train={}, test={}", train_count, test_count);
        }
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    }
}
