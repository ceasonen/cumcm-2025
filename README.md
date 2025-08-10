### **阶段一：环境安装 (每个团队成员都必须完成)**

这一步是基础，团队里的**每个人**都需要在自己的电脑上完成配置。
这里是B站的教程，提供给大家：https://www.bilibili.com/video/BV1y8411P7qs/?spm_id_from=333.337.search-card.all.click&vd_source=b4eb6038e60fc2b65b9647651ed4381f

#### **第1步：安装 LaTeX 发行版 (核心引擎)**

你的电脑需要一个能“解释”并“编译” LaTeX 代码的程序。VS Code 只是一个编辑器，真正干活的是它。

  * **推荐**: **TeX Live** (约 4-5 GB)
  * **下载地址**: [https://www.tug.org/texlive/acquire-iso.html](https://www.tug.org/texlive/acquire-iso.html)
  * **操作指南**:
    1.  下载 ISO 镜像文件。这会很慢，请耐心等待。
    2.  加载 ISO 文件 (Windows 10/11 和 macOS可以直接双击打开)。
    3.  运行里面的安装脚本 (`install-tl-windows.bat` 或 `install-tl`)。
    4.  **重要提示**: 在安装选项中，选择**完整安装 (full scheme)**。虽然占用空间大，但这可以确保你以后不会因为缺少某个宏包而编译失败，在数模竞赛这种争分夺秒的场景下至关重要。
    5.  安装过程可能需要半小时到一小时。

> **对于 Mac 用户**: 你也可以选择安装 **MacTeX** ([https://www.tug.org/mactex/](https://www.tug.org/mactex/))，它本质上就是为 macOS 包装好的 TeX Live。

#### **第2步：安装 VS Code (编辑器)**

  * **下载地址**: [https://code.visualstudio.com/](https://code.visualstudio.com/)
  * **操作指南**: 下载对应你操作系统的版本，一路“下一步”安装即可。(应该都安装过了，这步可以省略)

#### **第3步：在 VS Code 中安装 LaTeX Workshop 插件**

这是将 VS Code 变身为 LaTeX 神器的关键。

1.  打开 VS Code。
2.  点击左侧边栏的 **扩展(Extensions)** 图标 (四个方块的图标，快捷键 `Ctrl+Shift+X`)。
3.  在搜索框中输入 `LaTeX Workshop`。
4.  找到它，点击 **Install**。

#### **第4步：安装 Git (版本控制工具)**

  * **下载地址**: [https://git-scm.com/downloads](https://git-scm.com/downloads)
  * **操作指南**: 下载并安装，同样一路“下一步”即可。（应该也下载过了）

#### **第5步：配置 Git (首次使用)**

安装完 Git 后，打开你的终端（Windows 可以用 `Git Bash` 或 `CMD`，Mac/Linux 用 `Terminal`），输入以下两条命令，配置你的身份信息。这会告诉 Git 每次提交是谁做的。

```bash
git config --global user.name "你的名字或昵称"
git config --global user.email "你的邮箱地址"
```

#### **环境验证**

1.  在电脑上新建一个文件夹，用 VS Code 打开它。
2.  新建一个文件，命名为 `test.tex`。
3.  复制以下代码进去：
    ```latex
    \documentclass{article}
    \usepackage{ctex}
    \begin{document}
    你好，世界！ My first local LaTeX document.
    \end{document}
    ```
4.  按下 `Ctrl+S` 保存。此时 VS Code 右上角应该会出现一个绿色的**播放按钮**图标。点击它旁边的**放大镜图标**（View LaTeX PDF）。
5.  如果 VS Code 能成功编译并在右侧窗口显示出 PDF，那么恭喜你，你的本地环境配置成功了！

-----

### **阶段二：创建项目 (由团队负责人操作)**

**这一步我已经完成了**

#### **第1步：在 GitHub/Gitee 上创建账号并新建仓库**

  * **推荐平台**：**GitHub** (国际通用) 或 **Gitee (码云)** (国内访问速度更快)。
  * **操作指南**：
    1.  注册一个账号。
    2.  点击 "New repository" (新建仓库)。
    3.  **仓库名称**: 建议使用项目相关的名称，如 `cumcm-2025`。
    4.  **权限**: **极其重要！一定要选择 `Private` (私有)！** 否则你们的代码会被所有人看到，可能导致竞赛成绩无效。
    5.  **添加 .gitignore**: 在下拉菜单中选择 `TeX`。这会自动生成一个文件，告诉 Git 忽略掉编译产生的临时文件（如 `.log`, `.aux`, `.pdf` 等），让你的仓库保持干净。

#### **第2步：克隆 (Clone) 仓库到本地**

1.  在你的 GitHub/Gitee 仓库页面，点击绿色的 "Code" 按钮，复制仓库的 URL (HTTPS 格式)。
2.  在你的电脑上，打开终端或 VS Code 的集成终端 (`Ctrl+` \` \`\`)。
3.  `cd` 到你想要存放项目的目录下，然后运行命令：
    ```bash
    git clone [你刚才复制的URL]
    ```
    这会在当前目录下创建一个和你的仓库同名的文件夹。

#### **第3步：邀请你的队友**

1.  在 GitHub/Gitee 的仓库页面，找到 "Settings" -\> "Collaborators" (协作者)。
2.  添加你队友的 GitHub/Gitee 用户名，他们会收到邀请邮件。

-----

### **阶段三：日常协作流程 (所有人都要掌握)**

这是最重要的部分，规定了团队如何高效、安全地协作。

#### **1. 获取项目 (除队长外的成员)**

当队长邀请你后，接受邀请。然后像队长一样，使用 `git clone [仓库URL]` 命令将项目克隆到你的本地电脑。**这个 `clone` 操作对于每个人来说，只需要做一次。**

#### **2. 每日工作循环 (核心)**

**规则 A：开始工作前，先拉取 (Pull)**

> **切记！每天在你开始写任何东西之前，第一件事永远是 `git pull`！**

这会把云端上队友们的最新修改同步到你的本地，可以最大程度避免冲突。
在 VS Code 中，你可以点击左下角状态栏的同步按钮，或者在终端中运行 `git pull`。

**规则 B：编写与修改**

  * 像平常一样在 VS Code 中编写你的 `.tex` 文件。
  * 随时保存 (`Ctrl+S`)，随时点击右上角的按钮编译和预览。这都只发生在你的本地电脑上，不会影响队友。

**规则 C：提交你的工作 (Commit & Push)**

当你完成了一个阶段性的工作（比如写完一小节，修复了一个错误），就应该把它提交到云端，让队友看到（记住在commit里标注修改了什么，或者添加注释）。

**使用 VS Code 的命令行界面**
```bash

# 1. 进入仓库目录 (如果还没进去的话)
cd path/to/your/cumcm-2025

# 2. 将所有改动添加到暂存区
git add .

# 3. 提交改动并写好说明
git commit -m "添加了几个初始的 LaTeX 文件"

# 4. 推送到远程服务器
git push origin main
```

#### **3. 关于冲突 (Merge Conflict)**

如果不小心同时修改了同一个文件的同一行，`git pull` 的时候就会发生“冲突”。

  * **不要慌！** VS Code 会非常清晰地在文件中标记出冲突区域。
  * 它会提供选项，如 "Accept Current Change" (保留你的修改)、"Accept Incoming Change" (接受你队友的修改)、"Accept Both Changes" (都保留)。
  * **最好的解决方法是**：立即和修改这部分代码的队友沟通，确认应该保留哪个版本，或者如何将两个版本的内容合并起来，然后手动修改文件，解决冲突后，再重新走一遍 `Commit` 和 `Push` 的流程。

-----

### 总结与小贴士

  * **工作流**: `pull` -\> 修改 -\> `stage` -\> `commit` -\> `push`。
  * **勤提交**: 不要憋着好几天才提交一次。完成一个小功能就提交一次。
  * **勤拉取**: 经常 `pull`，保持本地版本最新。
  * **写好说明**: 每次 `commit` 都写清楚你干了什么。
  * **沟通**: 在修改关键部分或公共文件前，和队友打声招呼。

