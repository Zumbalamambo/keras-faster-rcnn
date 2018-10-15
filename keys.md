### １、python optparse模块之OptionParser  
***[原博客](https://blog.csdn.net/m0_37717595/article/details/80603884)***  

(1) 介绍   
Optparse，它功能强大，而且易于使用，可以方便地生成标准的、符合Unix/Posix 规范的命令行说明  
OptionParser是python中用来处理命令行的模块，在我们使用python进行流程化开发中必要的工具  

(2) 基本用法  
- 导入optionparser, 构造对象　parser = OptionParser()   
- 增加option, parser.add_option()  
每个命令行参数就是由**参数名字符串和参数属性**组成  

- 调用optionparser的解析函数： options, args = parser.parse_args()  
    - optParser.parse_args() 分析并返回一个字典和列表
    - options 是一个字典，  
    key 是所有的add_option()函数中的dest参数值，  
    value 是add_option()函数中的default的参数或者是由用户传入optParser.parse_args()的参数  
    - args, 是一个由 positional arguments 组成的列表。


(3) 实践
~~~
from optparse import OptionParser
optParser = OptionParser()

optParser.add_option('-f','--file',action = 'store',type = "string" ,dest = 'filename')
optParser.add_option("-v","--vison", action="store_false", dest="verbose", default='hello',help="make lots of noise [default]")

fakeArgs = ['-f','file.txt','-v','how are you', 'arg1', 'arg2']
option , args = optParser.parse_args()
op , ar = optParser.parse_args(fakeArgs)
print("option : ",option)
print("args : ",args)
print("op : ",op)
print("ar : ",ar)
--------------------------------------------------
# output
option :  {'filename': None, 'verbose': 'hello'}
args :  []
op :  {'filename': 'file.txt', 'verbose': False}
ar :  ['how are you', 'arg1', 'arg2']
~~~
     

(4) add_option()  
    add_option()参数说明：  
        action:存储方式，分为三种store、store_false、store_true  
        type:类型  
        dest:存储的变量  
        default:默认值  
        help:帮助信息  
  
### 2、sdssdsdssdssdsd