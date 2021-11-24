# Linux (Ubuntu) Commands

[TOC]

## I/O 文件和目录操作

### Extract *.7z
```shell
# install p7zip
sudo apt install p7zip
# extract 7z file
p7zip -d something.7z
```

### Extract/zip a folder
```shell
zip -r xxx.zip ./*
unzip filename.zip
tar -zxf XXX.tar.gz     -C 解压位置
```

### Folder Size 查看目录的大小
```shell
du -h --max-depth=1
```

### Count files 大量文件的个数统计（超过ls的限制时）
```shell
find -type f -name '*.mp4' | wc -l
```

### Split and merge files
```shell
split --bytes=1GB     /path/to/file.ext /path/to/file/prefix
cat prefix* >     file.ext
```

## Tools 效率工具

### Screen
Create a virtual screen:
```shell
screen -S [name]
```
- Exit screen: `Ctrl+A, D`
- Scroll: `Ctrl+A, Escape`

### Search history
```shell
history | grep 'something'
```

### Find & kill a process 关闭残留进程
```shell
ps -elf | grep python
kill -9 pid
```

### Monitor GPU utilization 监督GPU使用率
```shell
watch -n 1 -d nvidia-smi
```

## Image & Video Operations 图片视频操作
### FFMPEG
Compress a video 视频压缩
```shell
ffmpeg -i [src] -r 25 -b 3.5M -ar 24000 -s 960x540 [dst]
```

### Image resolution 查看图片的分辨率信息
```shell
file [filename]
```

### Del small-size images 删除所有的小于10k的jpg图
```shell
find . -name "*.jpg" -type 'f' -size -10k -delete
```
