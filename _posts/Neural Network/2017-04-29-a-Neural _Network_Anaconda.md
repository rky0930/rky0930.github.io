---
layout: post
title: "Neural Networks  1 - 1 | Anacodna"
date: 2017-4-29 15:33:00
comments: true
description: "Anaconda"
keywords: ""
categories:
- Neural_Network
tags:
- Anaconda
---
*이 포스트는 [Udacity - DLFND](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)를 강의 내용을 필기 형식으로 작성 했습니다.*

# What is Anaconda ? 
아나콘다는 파이썬의 패키지&환경 관리 도구 입니다. 파이썬으로 프로젝트를 진행하다 보면 시간이 지날 수록 환경과 설치한 패키지가 많아지는데요.. 나중엔 내가 뭘깔았는지 기억도 안납니다. 
근대 아나콘다는 설치한 패키지와 환경들을 분리하도록 해주기 때문에 기분이 이상하다 싶으면 지우고 다시 깔면 됩니다. 

# Installing Anaconda
여기서 다운로드 하시고요 
[https://www.continuum.io/downloads](https://www.continuum.io/downloads) 

업그레이드도 한번 해줍니다.
{% highlight bash %}
conda upgrade conda
conda upgrade --all
{% endhighlight %}


# Managing environments
아나콘다 환경 관리 방법 입니다. 기본적으로 환경 만들고, 지우고, 리스트 보고 할 수 있습니다. 
그리고 환경 만들면서 파이썬 버전을 선택하고, 사용 할 패키지도 설치할 수 있어요. 참 편하죠!
{% highlight bash %}
# 환경 만들기
conda create -n env_name list of packages
# ex) conda create -n tf_env tensorflow numpy pandas
# 파이썬 버전 선택해서 환경 만들기
conda create -n env_name python=version
# ex) conda create -n py3 python=3.5
# ex) conda create -n py2 python=2.7
{% endhighlight %}

생성 됬는지 조회한번 해주고 
{% highlight bash %}
# 환경 조회하기 
conda env list
{% endhighlight %}
쓰다가 환경이 너무 많거나 뭐깔았는지 기억도 안날땐 환경하나 새로 파는데요.
필요없는것 지우는건 아래와 같습니다.
{% highlight bash %}
# 환경 지우기
conda env remove -n env_name
{% endhighlight %}
환경 만들고 불러와서 써야죠. 
환경에 들어가지면 쉘 앞에 **(env_name)** 이렇게 바뀌는걸 확인 할 수 있습니다.
{% highlight bash %}
# 환경 불러오기 
source activate my_env
# ex) source activate tf_env
# 환경 끄기 : 환견 접속 상태에서 사용
source deactivate 
{% endhighlight %}

# Managing Packages
아나콘다는 파이썬 패키지 관리도 할 수 있습니다. 
설치하고 지우고 목록보고 등등명령어는 아래와 같습니다. 

{% highlight bash %}
# 설치
conda install package_name        
# ex) conda install numpy  
# 패키지 여러개 설치  
conda install package_name1 package_name2 package_name3
# ex) conda install numpy scipy pandas

# 설치 with 버젼 
conda install package_name=version
# ex) conda install numpy=1.10

# 삭제
conda remove package_name
# ex) conda remove numpy

# 업데이트 
conda update package_name
# ex) conda update numpy

# 모두 업데이트
#ex)conda update --all
{% endhighlight %}

# More environment actions

환경 설정한것 옮기고 싶으시면 설치리스트를 export한뒤 환경 생성할때 사용 할 수 있습니다.  
소스코드 공유할때도 환경 같이 공유해주시면 Good Job 입니다. 
{% highlight bash %}
# 파일로 출력 
conda env export > environment.yaml
# pip로 설치한것만 exprot 하려면 
pip freeze > requirements.txt 
# 파일로 부터 환경 생성
conda env create -f environment.yaml
{% endhighlight %}
