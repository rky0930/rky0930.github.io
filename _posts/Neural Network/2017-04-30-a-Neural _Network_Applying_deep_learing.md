---
layout: post
title: "Neural Networks  1 - 3 | Applying Deep Learning"
date: 2017-4-30 00:33:00
comments: true
description: "Applying Deep Learning"
keywords: ""
categories:
- Neural_Network
tags:
- Applying Deep Learning
---
*이 포스트는 [Udacity - DLFND](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)를 강의 내용을 필기 형식으로 작성 했습니다.*

# Applying Deep Learning
딥러닝 공부 시작전에 어디에서 딥러닝이 사용 되었는지 재밌는 예제를 보고 시작하세요.  
<br />

### Style Transfer
그림의 스타일을 학습시켜 사진과 합성 시켜줍니다.   
실행 시키지 전에 소스코드는 아래에서 받으세요.  
[fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)  

{% highlight bash %}

# 환경 생성 
conda create -n style-transfer python=2.7.9
source activate style-transfer
pip install tensorflow
conda install scipy pillow

# 실행 명령어 
python evaluate.py --checkpoint ./rain-princess.ckpt --in-path <path_to_input_file> --out-path ./output_image.jpg
{% endhighlight %}
<br />

### Deep Traffic

딥러닝으로 교통 시뮬레이션  
65mph 이상 받아보기 .. 모르겠다.  
[Deep Traffic](http://selfdrivingcars.mit.edu/deeptrafficjs/)  
<br />

### Flappy Bird
{% highlight bash %}
# 환경 생성 
conda create --name=flappybird python=2.7
# 환경 불러오기 
Mac/Linux: source activate flappybird
# 사용하는 패키지
conda install -c menpo opencv3
pip install pygame
pip install tensorflow
# 소스코드 다운로드 
git clone https://github.com/yenchenlin/DeepLearningFlappyBird.git
# 실행 
cd DeepLearningFlappyBird
python deep_q_network.py
{% endhighlight %}

