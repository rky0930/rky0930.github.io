---
layout: post
comments: true
categories: diary
---
## GPU 할당량 조절

# GPU 전체를 할당하지 말고 일단 필요한 만큼 올린다음 더 필요하면 점차적으로 늘리기
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config, ... )

# GPU 전체 메모리의 특정 퍼센트만 할당하기
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config, ... )

[출처] GPU 할당량 조절|작성자 wonkonge
