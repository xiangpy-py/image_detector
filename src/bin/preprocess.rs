use clap::Parser;
use image::{imageops::FilterType, ImageReader, ImageError};
use ndarray::{s, Array3, Array4};
use ndarray_npy::write_npy;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(about = "将胸部 X 光 JPEG 图像预处理为 .npy 缓存")]
struct Args {
    #[arg(long, env = "DATASET_ROOT", default_value = "/root/autodl-tmp/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray")]
    root: PathBuf,

    #[arg(long, env = "CACHE_DIR", default_value = "cache")]
    out: PathBuf,

    #[arg(long, default_value_t = 224)]
    size: u32,

    #[arg(long, value_delimiter = ',', default_values = ["0.485", "0.456", "0.406"])]
    mean: Vec<f32>,

    #[arg(long, value_delimiter = ',', default_values = ["0.229", "0.224", "0.225"])]
    std: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let size = args.size as usize;

    if args.mean.len() != 3 || args.std.len() != 3 {
        eprintln!("错误：--mean 和 --std 必须各包含 3 个浮点数");
        std::process::exit(1);
    }

    fs::create_dir_all(&args.out)?;

    let splits = [("train", "train"), ("test", "test")];

    for (split_dir, split_name) in &splits {
        let mut image_paths: Vec<(PathBuf, i64)> = Vec::new();
        for (class_name, label) in [("NORMAL", 0), ("PNEUMONIA", 1)] {
            let class_dir = args.root.join(split_dir).join(class_name);
            if !class_dir.exists() {
                eprintln!("警告：目录不存在 {:?}", class_dir);
                continue;
            }
            for entry in WalkDir::new(&class_dir)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if ext == "jpg" || ext == "jpeg" || ext == "png" {
                            let file_name = path.file_name().unwrap().to_string_lossy();
                            if !file_name.starts_with('.') {
                                image_paths.push((path.to_path_buf(), label));
                            }
                        }
                    }
                }
            }
        }

        image_paths.sort_by(|a, b| a.0.cmp(&b.0));
        let n = image_paths.len();
        println!("{}: 找到 {} 张图像", split_name, n);

        let mut images = Array4::<f32>::zeros((n, 3, size, size));
        let mut labels = ndarray::Array1::<i64>::zeros(n);

        for (i, (path, label)) in image_paths.iter().enumerate() {
            let img = load_and_preprocess(path, args.size, &args.mean, &args.std)?;
            images.slice_mut(s![i, .., .., ..]).assign(&img);
            labels[i] = *label;

            if (i + 1) % 500 == 0 || i + 1 == n {
                println!("{}: 已处理 {}/{}", split_name, i + 1, n);
            }
        }

        let images_path = args.out.join(format!("{}_images.npy", split_name));
        let labels_path = args.out.join(format!("{}_labels.npy", split_name));

        write_npy(&images_path, &images)?;
        write_npy(&labels_path, &labels)?;

        println!("已保存: {:?} 和 {:?}", images_path, labels_path);
    }

    Ok(())
}

fn load_and_preprocess(
    path: &Path,
    size: u32,
    mean: &[f32],
    std: &[f32],
) -> Result<Array3<f32>, ImageError> {
    let img = ImageReader::open(path)?.decode()?;
    let img = img.resize_exact(size, size, FilterType::Lanczos3).to_rgb8();

    let mut tensor = Array3::<f32>::zeros((3, size as usize, size as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        for c in 0..3 {
            let val = pixel[c] as f32 / 255.0;
            tensor[[c, y as usize, x as usize]] = (val - mean[c]) / std[c];
        }
    }

    Ok(tensor)
}
