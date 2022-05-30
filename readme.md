## AIprogramming project
##### 사용명령어
> python main.py --data_num 3

##### data폴더 생성하고 내부에 dataset1-5 넣고 돌려야함

- 다른 data_num에 대한 모델 최적화는 아직 확인 안해봄. 일단 3은 잘돌아감
- PCA 사용 시 do_pca_lreg에서 원하는 component 개수로 n_components 설정 -> 80이 최적값임


* cnn 에러 (dataset1)

ytrain = keras.utils.to_categorical(ytrain, num_classes)

File "/opt/conda/lib/python3.8/site-packages/tensorflow/python/keras/utils/np_utils.py", line 74, in to_categorical

categorical = np.zeros((n, num_classes), dtype=dtype)

TypeError: 'numpy.float64' object cannot be interpreted as an integer
