use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::cache::save_split_cache;
use crate::dataset::scan_split;
use crate::image::load_and_resize_rgb;

/// 处理一个 split：扫描目录、并行加载缩放图像、过滤损坏图像、写入缓存。
///
/// `merge_val` 仅当 `split_name == "train"` 时生效：true 则同时扫描 val 目录。
///
/// 返回有效图像数量。
pub fn process_split(
    root: &Path,
    out: &Path,
    size: u32,
    split_name: &str,
    merge_val: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    let entries = scan_split(root, split_name, merge_val);
    let total = entries.len();
    println!("{}: 找到 {} 张图像", split_name, total);

    if total == 0 {
        return Ok(0);
    }

    let processed_count = AtomicUsize::new(0);

    let results: Vec<_> = entries
        .par_iter()
        .map(|entry| {
            let result = match load_and_resize_rgb(&entry.path, size) {
                Ok(img) => Ok(img),
                Err(e) => {
                    eprintln!("警告：跳过损坏/无法读取的图像 {:?}: {}", entry.path, e);
                    Err(format!("{}", e))
                }
            };
            let cnt = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
            if cnt % 500 == 0 || cnt == total {
                println!("{}: 已处理 {}/{}", split_name, cnt, total);
            }
            (result, entry.label)
        })
        .collect();

    let valid_results: Vec<_> = results
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

    let (images, labels): (Vec<_>, Vec<_>) = valid_results.into_iter().unzip();
    save_split_cache(out, split_name, size, &images, &labels)?;

    Ok(n)
}
