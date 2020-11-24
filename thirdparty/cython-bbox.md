# FairMOT | win10下cython-bbox安装的心酸之路
 最近的MOT杀出了一匹黑马[FairMOT](https://zhuanlan.zhihu.com/p/131430303)，于是我心痒难耐想拿来试试，我是在自己笔记本上跑的，但是安装环境的时候cython-bbox一直安装报错，作为一只初级菜鸟只能到处找博客解决，但是找了一天都没解决，甚至还重装了vs2015，我也不知道有没有用，后来终于发现了这个，分分钟给弄好了，哎心累~  这里分享给大家一起看看，其实挺简单的。【心疼的抱住渣渣的自己】

## 问题
	pip install cython-bbox
这里可能回报错：no moddle of cython，安装cython模块就好了：	pip install cython 
然后再  pip install cython-bbox

结果报错：
cl: 命令行 error D8021 :无效的数值参数“/Wno-cpp” 
 error: command 'D:\\programs\\vision studio 2015\\VC\\BIN\\x86_amd64\\cl.exe' failed with exit status 2
 ERROR: Failed building wheel for cython-bbox

## 解决办法
1.下载[cython-bbox](https://pypi.org/project/cython-bbox/)

2.解压文件

3.找到steup.py 文件，把extra_compile_args=['-Wno-cpp'],修改为extra_compile_args = {'gcc': ['/Qstd=c99']}

4.在解压文件目录下运行

	python setup.py build_ext install
![安装结果](https://img-blog.csdnimg.cn/2020052915565469.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDkxMjk4Nw==,size_16,color_FFFFFF,t_70#pic_center)
出现这样的结果就可以了，可以去环境中查看，cython-bbox已在annaconda环境中。