---
layout: post
title: "Neural Networks  1 - 2 | Jupyter Notebook"
date: 2017-4-29 20:33:00
comments: true
description: "Jupyter Notebook"
keywords: ""
categories:
- Neural_Network
tags:
- Jupyter Notebook
---
*이 포스트는 [Udacity - DLFND](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)를 강의 내용을 필기 형식으로 작성 했습니다.*

# What are Jupyter notebooks?
설치하면 Web App이 동작합니다.
브라우져로 웹서비스에 원격 및 로컬에서 접속하여 코딩 할 수 있습니다.
코드를 조각 조각 Cell로 나누어 실행 할 수 있어 코딩하기 편리합니다.
사진, 그래프, markdown등 을 사용하여 코드 설명도 남길 수 있습니다.
파일 확장자는 .ipynb 를 사용 합니다. 

# Installing Jupyter Notebook
설치 방법 입니다.  
설치시에 conda로 환경 설정하고 설치하시면 해당 환경에서 설치 됩니다.

{% highlight bash %}
# Conda를 사용하여 설치 
conda install jupyter notebook
# pip 사용하여 설치 
pip install jupyter notebook
{% endhighlight %}

# Launching the notebook server
실행 방법은 그냥 쉘에 jupyter notebook 을 쓰면 됩니다.  
실행 시킨 쉘에서 환경이 파이썬 노트북의 환경이 됩니다.  

{% highlight bash %}
# 실행 방법 
jupyter notebook
{% endhighlight %}