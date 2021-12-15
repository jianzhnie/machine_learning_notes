# Git  的常规操作

[TOC]

## 常规操作

### Clone a Repo (在线创建仓库)

```shell
git clone git@github.com:jianzhnie/deep_head_pose.git
cd deep_head_pose
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

### Create Repo from Local folder (本地创建仓库)

```shell
##  create a new repository on the command line
echo "# deep_head_pose" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin git@github.com:jianzhnie/deep_head_pose.git
git push -u origin master
```
### Push a Local Repo 上传已经存在的仓库
```shell
## push an existing repository from the command line
git remote add origin git@github.com:jianzhnie/deep_head_pose.git
git push -u origin master
```

### Branch 分支操作

``` shell
# 列出所有本地分支
$ git branch

# 列出所有远程分支
$ git branch -r

# 列出所有本地分支和远程分支
$ git branch -a

# 新建一个分支，但依然停留在当前分支
$ git branch [branch-name]

# 新建一个分支，并切换到该分支
$ git checkout -b [branch]

# 新建一个分支，指向指定commit
$ git branch [branch] [commit]

# 新建一个分支，与指定的远程分支建立追踪关系
$ git branch --track [branch] [remote-branch]

# 切换到指定分支，并更新工作区
$ git checkout [branch-name]

# 切换到上一个分支
$ git checkout -

# 建立追踪关系，在现有分支与指定的远程分支之间
$ git branch --set-upstream [branch] [remote-branch]

# 合并指定分支到当前分支
$ git merge [branch]

# 选择一个commit，合并进当前分支
$ git cherry-pick [commit]

# 删除分支
$ git branch -d [branch-name]

# 删除远程分支
$ git push origin --delete [branch-name]
$ git branch -dr [remote/branch]
```

### Commits 操作

#### git reset - 撤销commits

Undo local commits, make the committed changes to unstaged status:

```shell
git reset --soft HEAD^
```

Discard all local uncommitted changes::warning:

```shell
git reset --hard
```

Discard all local unpushed changes: :warning:

```shell
git reset --hard @{u}
```

#### git stash - 保存于恢复工作区

```shell
# Save working copy
git stash
# restore working copy
git stash pop
```

#### git rebase - 合并commits

Merge commits inside a branch

```shell
git rebase -i [startpoint] [endpoint] （前开区间、后闭区间）
```
Merge commits from current branch to another branch

```shell
git rebase  [startpoint]  [endpoint] --onto [branchName]
```

#### git submodules

```shell
# Add a submodule to a repo
git submodule add <url> <name>
git add <name>
git commit -m "example comments"
git push

# Pull a repo with its submodules
git pull --recurse-submodules
```


### Alias 简化命令

```shell
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status

# then you can use "git co" instead of "git checkout", for example:
git co master
```



## 高级操作

### 保持 forked repository 和官方仓库保持同步更新

例如，我最近 fork 了 `mmdetection` 官方仓库到我的 github 地址， 修改了部分文件，并且 push 到我的 github 上。过了一段时间， `mmdetection` 官方仓库有了新的更新， 但是我 fork 的版本没有包含进来，因此我该如何保持我维护的 `mmdetection` 和官方版本保持同步？

> In your local clone of your forked repository, you can add the original GitHub repository as a "remote". ("Remotes" are like nicknames for the URLs of repositories - origin is one, for example.) Then you can fetch all the branches from that upstream repository, and rebase your work to continue working on the upstream version. In terms of commands that might look like:


#### Syncing a fork

**Step 1**

```shell
git clone https://github.com/open-mmlab/mmdetection.git

## use git remote to see the origin url
git remote -v
origin	https://github.com/open-mmlab/mmdetection.git (fetch)
origin	https://github.com/open-mmlab/mmdetection.git (push)
```

**Step 2**

```shell
## second, change the origin url
git remote set-url origin https://github.com/apulis/ApulisVision.git

## make sure you have modified successful
git remote -v
origin	git@github.com:apulis/ApulisVision.git (fetch)
origin	git@github.com:apulis/ApulisVision.git (push)
```

**Step 3**
Before you can sync, you need to add a remote that points to the upstream repository. You may have done this when you originally forked.

```shell
# Add the remote, call it "upstream":

git remote add upstream https://github.com/open-mmlab/mmdetection.git

## make sure you have modified successful
git remote -v
origin	git@github.com:apulis/ApulisVision.git (fetch)
origin	git@github.com:apulis/ApulisVision.git (push)
upstream	https://github.com/open-mmlab/mmdetection.git (fetch)
upstream	https://github.com/open-mmlab/mmdetection.git (push)
```

####  Fetching
There are two steps required to sync your repository with the upstream: first you must fetch from the remote, then you must merge the desired branch into your local branch.

Fetching from the remote repository will bring in its branches and their respective commits. These are stored in your local repository under special branches.

```shell
git fetch upstream

# Grab the upstream remote's branches
remote: Counting objects: 75, done.
remote: Compressing objects: 100% (53/53), done.
remote: Total 62 (delta 27), reused 44 (delta 9)
Unpacking objects: 100% (62/62), done.
From https://github.com/open-mmlab/mmdetection
 * [new branch]      master     -> upstream/master
```

We now have the upstream's master branch stored in a local branch, upstream/master

```shell
git branch -va
* master                  22d2612 Merge remote-tracking branch 'upstream/master'
  remotes/origin/HEAD     -> origin/master
  remotes/origin/master   22d2612 Merge remote-tracking branch 'upstream/master'
  remotes/upstream/master 7f0c4d0 fix sampling result method typo (#3224)
```

####  Merging

Now that we have fetched the upstream repository, we want to merge its changes into our local branch. This will bring that branch into sync with the upstream, without losing our local changes.

```shell
$ git checkout master
# Check out our local master branch
Switched to branch 'master'

$ git merge upstream/master
# Merge upstream's master into our own
Updating a422352..5fdff0f
Fast-forward
 README                    |    9 -------
 README.md                 |    7 ++++++
 2 files changed, 7 insertions(+), 9 deletions(-)
 delete mode 100644 README
 create mode 100644 README.md
```

If your local branch didn't have any unique commits, git will instead perform a "fast-forward":

```shell
$ git merge upstream/master
Updating 34e91da..16c56ad
Fast-forward
 README.md                 |    5 +++--
 1 file changed, 3 insertions(+), 2 deletions(-)
```
