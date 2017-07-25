---
layout: post
comments: true
categories: diary
---
## Python logger

{% highlight python %}
import logging

logger = logging.getLogger("yoon")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)
def aaa(test):
    print(test)
    logger.debug(test) 


aaa('test')
'''
output: [detect.py:15 - aaa() ] test
'''
{% endhighlight %}
