---
layout: post
title: "Neural Networks  1 - 5 | Intro to Neural Networks
"
date: 2017-5-7 01:45:00
comments: true
description: "Intro to Neural Networks"
keywords: ""
categories:
- Intro_to_Neural_Networks
tags:
- Intro to Neural Networks
published: True
---

Udacity의 Deep Learing 강의에서 "Intro to Neural Networks"은  Luis Serrano 라는 분이 맡았는데요. 이분의 강의는 [github](https://github.com/AccelAI/DemystifyingDeepLearning-AI/blob/master/nov16/november16.md)와 [YouTube](https://youtu.be/BR9h47Jtqyw)에 공개 되어 있습니다. 저도 공개된 자료를 참고하여 포스팅 하려고 합니다.  
<br />

# Intro to Neural Networks

1. Neural Networks를 상상 할때 아래와 같은 그림을 떠오른적 있으신가요 ?  멋진 로봇 & 뇌, 그리고 외계인 등등
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Intro_1.png" width="600">
<br /><br />

2. 실제 Neural Networks는 아래와 같습니다.  
못생겼죠..? 복잡해 보이네요. 저게 뭔지도 모르겠고 .. ㅎㅎ 저도 같은 생각 이었습니다. 너무 걱정하지 마세요.  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Intro_2.png" width="800">
<br /><br />

3. 이 포스팅이 참고하는 자료에 저자 Luis Serrano는 사실 Neural Networks의 목적을 데이터 분류라고 생각 한다고 하네요. 
아래 사진 처럼 해변에 선을 그어 "빨간 조개" 와 "파란 소라"를 분류 하는 것과 같다고 합니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Intro_3.png" width="600">
<br />
Neural Networks에 간략하게 알아보기 전에 2 가지를 먼저 설명 하겠습니다.  
 - Logistic Regression  
 - Gradient Descent  
본론으로 들어가기 전에 조금만 참으시고 간단한 설명을 보고 지나가시길 바랍니다.  

<br />

# Logistic Regression  
1. 물음표 위치에 시험점수와 등급을 받은 학생합격/탈락을 어떻게 결정하면 될까요 ?  
: Logistic Regression을 사용하여 최적의 기준선을 찾고, 기준선에 따라 결과를 예측하면 됩니다.  
 - 기준선  위에 존재하는 점 => 합격  
 - 기준선 아래에 존재하는 점 => 탈락
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_1.png" width="800">
(그래프 설명 1) 아래의 표는 시험점수와 등급을 2차원 그래프에 표시  
(그래프 설명 2) 파란점은 대입합격을 빨간점은 대입 탈락을 표시
<br /><br />

2. 최적의 기준선을 찾는 법을 단계별로 설명 해보겠습니다.  
기준선을 Random하게 설정해 보겠습니다.  
2개 에러가 발생했네요. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_2.png" width="800">
<br /><br />
3. 조금 더 괜찮게 기준선을 정해볼까요?  
데이터 분류선을 조금 움직여 보겠습니다.  
1개 에러가 발생했네요. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_3.png" width="800">
<br /><br />
4. 데이터 분류선을 조금 더 움직여 보겠습니다.  
드디어 0개 에러가 발생하게 되었습니다.  
<br />
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_4.png" width="800">
Logistic Regression에서도 이와 같이 에러가 조금씩 줄드는 방향으로 기준선을 이동 시키며 최적의 위치를 찾아냅니다.  
<br /><br />
5. 조금 더 자세하게 설명해 보겠습니다.  
지금 부터는 에러를 점수로 표시하려고 합니다.  
점수 계산 법은 아래와 같습니다.  
 - 기준선에 따라 점이 잘~ 분류된 경우 낮은 에러 점수 추가
 - 기준선에 따라 점이 잘못 분류된 경우 높은 에러 점수 추가  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_5.png" width="800">
6. 따라서, 잘못 분류 된다면 에러점수가 올라가고, 잘 분류되면 에러 점수가 내려갑니다.
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_6.png" width="800">
<br /><br />
7. 에러 점수 가장 낮은 분류선을 찾는 것은 최적의 기준선을 찾은 것 입니다. 
어떻게 점점 에러 점수를 낮게 받을 수 있을까요 ?
<br />
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_7.png" width="800">
: 바로 [Gradient Descent](https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95) 입니다.  

8. [Gradient Descent](https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95) 는 산에서 내려오는 것에 비유 할 수 있습니다.    
"내려가는 방향으로 계속 이동" = "에러를 줄 일 수 있도록 기준선을 이동"  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Gradient_Descent_1.png" width="800">
<br /><br />
9. 반복해서, 에러를 줄 일 수 있도록 기준선을 옮깁니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Gradient_Descent_2.png" width="800">

<br /><br />
10. 마침내, 최소의 에러점에 도착합니다.
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Gradient_Descent_3.png" width="800">

<br /><br />
11. [Gradient Descent](https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95)의 기본 아이디어는 "에러 함수(Error Function)"의 기울기(경사)를 구하고 기울기가 낮은 쪽으로 계속 이동시켜서 최적의 값을 구하는 것 입니다.  
** 주의: 에러 함수의 기울기 입니다. 기준선의 기울기가 아닙니다. 
<br />
<br />
Error Function은 분류된 상태의 에러를 수치화하는 함수 입니다.  
 예) 잘 분류하면 낮은 점수, 잘 못 분류하면 높은 점수  
어떻게 만들 수 있을까요 ? 간단하게 설명 해보겠습니다.  
<br />
아래 사진은 그래프 전체의 확률을 표시한 것 입니다.   
- 우측 상단의 진한 파랑 부분은 파란점이 위치할 확률이 아주 높은 부분  
- 좌측 하단의 진한 빨강 부분은 빨강점이 위치할 확률이 아주 높은 부분   
- 기준선 쪽으로 이동하면서 확률은 50:50 입니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_8.png" width="800">
12. 수치화된 확률을 구하기 위해 기준선에서 일정한 긴갹마다 점수를 부여합니다.  
 - 파랑 부분은 + 가 부여 됩니다.   
 - 빨강 부분은 - 가 부여 됩니다.  
이렇게 되면 기준에서 거리가 먼 부분은 +- 로 큰 수가 부여됩니다.
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_9.png" width="800">
13. 부여된 점수를 확률로 만들기 위해서 우리가 사용할 [Activation Function](https://en.wikipedia.org/wiki/Activation_function)은 [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) 입니다.   
 아래 사진에서 보이듯이 [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) 함수는  
 - 음수는 0에 가깞게 만듭니다. 예) -10 = 0.0001  
 - 양수는 1에 가깝게 만듭니다. 예)   5 = 0.99  
 - 0 = 0.5  

<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_10.png" width="800">
<br /><br />
14. 우리가 원하는 그래프 전체의 확률을 알 수 있게 되었습니다.  
이제 확률로 Error를 계산해 보겠습니다.  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_11.png" width="800">
<br /><br />
<br /><br />
15. 각 점에서 확률을 구하면 됩니다.  
주의하실 점은 위치 뿐만 아니라 점에 색에 따라 확률이 달라집니다.  
 예1) 파란 부분에서 파란 점이 나올 확률 높음  
 예2) 파란 부분에서 빨간 점이 나올 확률 낮음  
 <br />
 각 점의 확률을 곱해서 에러 점수를 구할 수 도 있지만,  
 잘 못 분류된 점에 패널티를 부여하기위해 각 확률에 - Log 함수를 취하겠습니다.  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Logistic_Regression_12.png" width="800">
16. 왼쪽 사진에서 높은 에러 점수가 표시되는 것을 확인 할 수 있습니다.  
우리가 원하는 Error Function이 완성 되었습니다.  
<br />
Neural Network의 목적은 분류라고 하였습니다.  
따라서, 우리는 최적의 분류선을 구하는 방법을 알아 봤습니다.  
이제 우리는 분류선을 사용하는 방법을 알아보고  
그래프에서 Neural Network 형태로 표현하는 방법을 알아보갰습니다.  
<br />
17. 기준선을 Nueron의 형태로 표현하는 방법은  
표현식을 Y = aX + b의 형태에서 w1X + w2Y = B 형태로 변경한 뒤  
오른쪽과 같이 표현 합니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_1.png" width="800">
<br /><br />
18. 이 표현 방식은 실제 Nueron의 모습과도 비슷 합니다.
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_2.png" width="800">
<br /><br />
19. 우리가 분류하려고 하는 데이터가 Non-Linear하면 어떻게 분류 할 수 있을까요 ?
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_3.png" width="800">
<br /><br />
20. 2개의 분류선을 사용 하면 됩니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_4.png" width="800">
<br /><br />
21. 두개 분류 결과를 더하여 표현하면 아래와 같은 영역의 분류가 가능합니다.  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_5.png" width="800">
<br /><br />
22. 첫번째 분류에서 나온 결과와 두번째 분류에서 나온 결과를 더하면
양수의 결과가 나와 파랑이라고 분류되는 것 입니다.
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_6.png" width="800">
<br /><br />
23. 각 분류의 결과에 가중치를 다르게 주어 최종 분류의 형태를 아래와 같이 바꿀 수도 있습니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_7.png" width="800">
<br /><br />
24. 첫번째 결과에 7의 가중치를 주고 두번째 결과에 5의 가중치를 부여한후 -6을 취하여 계산합니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_8.png" width="800">
<br /><br />
25. 기준선을 Nueron의 형태로 표현 했듯이,  
Neuron을 연결하여 Neural Network 형태로 표현 할 수 있습니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_9.png" width="800">
<br /><br />
26. 우선 처음 Linear하게 분류한 그래프를 Neuron의 형태로 표현합니다.  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_10.png" width="800">
<br /><br />
27. Neuron을 연결하고 가중치도 표현하여 Neural Network로 만듭니다.
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_11.png" width="800">
<br /><br />
28. 각 그래프의 입력 부분을 같이 표현하여 좀 더 간결하게 수정할 수 있습니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_12.png" width="800">
<br /><br />
29. Nueral Network는 Layer로 분류하여 말 할 수 있습니다.  
처음 입력 부분을 Input Layer 라고 합니다.  
마지막에 출력 부분을 Output Layer 라고 합니다.  
그리고, 그 사이 모든 Layer를 Hidden Layer라고 합니다
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_13.png" width="800">
<br /><br />
30. 3개의 Nueron을 연결하면 아래와 같이 분류 할 수 있습니다. 
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_14.png" width="800">
<br /><br />
31. 입력값을 3개로 하면 3차원의 분류도 가능합니다. ^^;
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_15.png" width="800">
<br /><br />
32. Hidden Layer의 2층으로 하여 더 복잡한 분류도 가능합니다.  
Hidden Layer가 2층 이상의 Neural Network는 Deep Neural Network라고 합니다.  
<img src="{{ site.url }}/assets/static/Intro_to_Neural_Network/Neural_Network_16.png" width="800">
<br /><br />

Neural Network를 소개해 봤습니다.    
이 글을 통해 여러분에게  
 - 데이터를 Linear 하게 분류하는 방법 (Logistic Regression)  
 - 최적의 분류선을 찾는 방법 (Gredient Descent)  
 - 각 그래프를 Neuron으로 표현하는 법  
 - Neuron을 연결하여 Neural Network로 만드는 법  
 - Neural Network를 사용하여 Non-Linear한 데이터를 분류하는 법  
을 소개 하였습니다. 
<br />
<br />
물론, 훨~씬 많은 과정이 필요하지만 간단하게 말하면   
- AlphaGo는 바둑판의 사진을 입력으로 받아 다음 수를 결정 합니다.  
(즉, 각 Pixel에 RGB로된 값을 입력으로 받아 다음 수를 어떻게 분류 한다고 할 수 있습니다.)  
- 자율주행 자동차는 Lidar의 데이터를 입력으로 받아 Left, Right, Forward, Reverse 4가지 방향 선택을 합니다.  
(즉, 센서값을 입력으로 받아 다음 주행 해야할 방향을 분류해 낸다고 할 수 있습니다.)
<br />
<br />
Luis Serrano가 사실 Neural Networks의 목적을 데이터 분류라고 하였습니다.  
생각해보니 실제 어떻게 잘~ 사용할 수 있는지 조금은 이해가 되는 것 같습니다.  
<br />
여기까지 읽어주셔서 감사합니다. 







