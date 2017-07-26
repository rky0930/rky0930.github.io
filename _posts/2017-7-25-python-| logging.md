---
layout: post
comments: true
categories: diary
---
## Python logger

{% highlight python %}
import logging

class GlobalLogging():
    logger = logging.getLogger("yoon")
    FORMAT = "[yoon_debug | %(filename)s:%(lineno)s - %(funcName)1s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.DEBUG)


'''
# import 
from yoon_logging import GlobalLogging

# use in code
GlobalLogging.logger.debug('Any debug message.') 

# output
[yoon_debug | show.py:176 - f_name() ] Any debug message.
'''
{% endhighlight %}
