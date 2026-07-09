use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{s, Array3, Array4};
use ndarray_npy::write_npy;
use pyo3::prelude::*;
use rayon::prelude::*;
use walkdir::WalkDir;

use image::{imageops::FilterType, ImageError, ImageReader};

fn load_and_resize(path: &Path, size: u32) -> Result<Array3<u8>, ImageError> {
    let img = ImageReader::open(path)?.decode()?;
    let img = img.resize_exact(size, size, FilterType::Lanczos3).to_rgb8();

    let mut tensor = Array3::<u8>::zeros((3, size as usize, size as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        for c in 0..3 {
            tensor[[c, y as usize, x as usize]] = pixel[c];
        }
    }

    Ok(tensor)
}

fn process_split(
    root: &Path,
    out: &Path,
    size: u32,
    split_name: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut image_paths: Vec<(PathBuf, i64)> = Vec::new();

    let source_dirs: Vec<&str> = if split_name == "train" {
        vec!["train", "val"]
    } else {
        vec![split_name]
    };

    for dir in &source_dirs {
        for (class_name, label) in [("NORMAL", 0), ("PNEUMONIA", 1)] {
            let class_dir = root.join(dir).join(class_name);
            if !class_dir.exists() {
                eprintln!("警告：目录不存在 {:?}", class_dir);
                continue;
            }
            for entry in WalkDir::new(&class_dir).into_iter().filter_map(|e| e.ok()) {
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
    }

    image_paths.sort_by(|a, b| a.0.cmp(&b.0));
    let total = image_paths.len();
    println!("{}: 找到 {} 张图像", split_name, total);

    if total == 0 {
        return Ok(0);
    }

    let processed_count = AtomicUsize::new(0);

    let results: Vec<(Result<Array3<u8>, String>, i64)> = image_paths
        .par_iter()
        .map(|(path, label)| {
            let result = match load_and_resize(path, size) {
                Ok(img) => Ok(img),
                Err(e) => {
                    eprintln!("警告：跳过损坏/无法读取的图像 {:?}: {}", path, e);
                    Err(format!("{}", e))
                }
            };
            let cnt = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
            if cnt % 500 == 0 || cnt == total {
                println!("{}: 已处理 {}/{}", split_name, cnt, total);
            }
            (result, *label)
        })
        .collect();

    let valid_results: Vec<(Array3<u8>, i64)> = results
        .into_iter()
        .filter_map(|(r, label)| r.ok().map(|img| (img, label)))
        .collect();

    let n = valid_results.len();
    let skipped = total - n;
    if skipped > 0 {
        println!("{}: 跳过 {} 张损坏/无法读取的图像", split_name, skipped);
    }
    println!("{}: 有效图像 {} 张，开始写入缓存...", split_name, n);

    if n == 0 {
        return Ok(0);
    }

    let sz = size as usize;
    let mut images = Array4::<u8>::zeros((n, 3, sz, sz));
    let mut labels = ndarray::Array1::<i64>::zeros(n);

    for (i, (img, label)) in valid_results.iter().enumerate() {
        images.slice_mut(s![i, .., .., ..]).assign(img);
        labels[i] = *label;
    }

    fs::create_dir_all(out)?;

    let images_path = out.join(format!("{}_images.npy", split_name));
    let labels_path = out.join(format!("{}_labels.npy", split_name));

    write_npy(&images_path, &images)?;
    write_npy(&labels_path, &labels)?;

    println!("已保存: {:?} 和 {:?}", images_path, labels_path);

    Ok(n)
}

/// 公共接口：处理数据集并生成 .npy 缓存。
///
/// # 参数
/// - `root`: 数据集根目录（包含 train/val/test 子目录）
/// - `out`: 缓存输出目录
/// - `size`: 图像目标尺寸（如 256）
///
/// # 返回
/// (训练集图像数, 测试集图像数)
pub fn preprocess_dataset_impl(
    root: &str,
    out: &str,
    size: u32,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let root = Path::new(root);
    let out = Path::new(out);

    let train_count = process_split(root, out, size, "train")?;
    let test_count = process_split(root, out, size, "test")?;

    Ok((train_count, test_count))
}

// ─── PyO3 Python 绑定 ───

#[pyfunction]
fn preprocess_dataset(root: &str, out: &str, size: u32) -> PyResult<(usize, usize)> {
    preprocess_dataset_impl(root, out, size)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn rust_preprocessor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_dataset, m)?)?;
    Ok(())
}
