use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// 数据集中的一个样本。
pub struct ImageEntry {
    pub path: PathBuf,
    pub label: i64,
}

/// 扫描指定 split 下的所有图像。
///
/// 当 `split_name` 为 "train" 且 `merge_val` 为 true 时，会同时扫描
/// "train" 和 "val" 目录（kaggle 数据集兼容行为）；
/// 否则只扫描 `split_name` 目录。
/// 每个目录下期望包含 "NORMAL" 和 "PNEUMONIA" 两个子目录。
pub fn scan_split(root: &Path, split_name: &str, merge_val: bool) -> Vec<ImageEntry> {
    let mut source_dirs: Vec<&str> = vec![split_name];
    if split_name == "train" && merge_val {
        source_dirs.push("val");
    }

    let mut entries = Vec::new();
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
                                entries.push(ImageEntry {
                                    path: path.to_path_buf(),
                                    label,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    entries.sort_by(|a, b| a.path.cmp(&b.path));
    entries
}
