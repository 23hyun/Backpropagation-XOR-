# 실제로 XOR 연산을 표현하는 parameter를 구하는 과정# 랜덤하게 초기화된 가중치가 XOR과 똑같이 조정되는 지 확인하자.

import random
import numpy as np

random.seed(777)

# 환경 변수 지정# 입력값 및 타겟값# XOR 연산의 진리표
data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

# 실행 횟수(iterations), 학습률(lr), 모멘텀 계수(mo) 설정
iterations=5000
lr=0.1
mo=0.4

# 활성화 함수 - 1. 시그모이드
# 미분할 때와 아닐 때의 각각의 값을 표현할 수 있게한다.
# 기본값은 미분이 아닐 때의 값
def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# 활성화 함수 - 2. tanh
# tanh 함수의 미분은 1 - (활성화 함수 출력의 제곱)
def tanh(x, derivative=False):
    if (derivative == True):
        return 1 - x ** 2
    return np.tanh(x)

# 가중치 배열 만드는 함수
def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
#[fill]을 j만큼 반복한다
return mat

# 신경망 클래스
class NeuralNetwork:

# 생성자를 이용해 초깃값의 지정
def __init__(self, num_x, num_yh, num_yo, bias=1):

# 입력값(num_x), 은닉층 초깃값(num_yh), 출력층 초깃값(num_yo), 바이어스self.num_x = num_x + bias# 바이어스는 1로 지정(본문 참조)
				self.num_yh = num_yh
        self.num_yo = num_yo

# 활성화 함수 초깃값
# 각자 노드 개수만큼 있다.
				self.activation_input = [1.0] * self.num_x
        self.activation_hidden = [1.0] * self.num_yh
        self.activation_out = [1.0] * self.num_yo

# 가중치 입력 초깃값. 랜덤하게 한다.
self.weight_in = makeMatrix(self.num_x, self.num_yh)
        for i in range(self.num_x):
            for j in range(self.num_yh):
                self.weight_in[i][j] = random.random()

# 가중치 출력 초깃값. 랜덤하게 한다.
self.weight_out = makeMatrix(self.num_yh, self.num_yo)
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()

# 모멘텀 SGD를 위한 이전 가중치 초깃값
self.gradient_in = makeMatrix(self.num_x, self.num_yh)
        self.gradient_out = makeMatrix(self.num_yh, self.num_yo)

# 업데이트 함수 : 입력층, 은닉층, 출력층의 모든 노드 값을 구한다.
def update(self, inputs):

# 입력 레이어의 활성화 함수까지 거친 결과값
# input에는 real data의 input이 들어간다.
# 입력층의 활성화 함수는 아무런 효과가 없다. 그냥 받은 거 그대로 내보낸다.
for i in range(self.num_x - 1):
            self.activation_input[i] = inputs[i]

# 은닉층의 활성화 함수까지 거친 결과값
for j in range(self.num_yh):
# 임시 변수
            sum = 0.0
# 가중합 구하기
for i in range(self.num_x):
                sum = sum + self.activation_input[i] * self.weight_in[i][j]
# 시그모이드와 tanh 중에서 활성화 함수 선택
self.activation_hidden[j] = tanh(sum, False)

# 출력층의 활성화 함수까지 거친 결과값
for k in range(self.num_yo):
# 임시 변수
            sum = 0.0
            for j in range(self.num_yh):
                sum = sum + self.activation_hidden[j] * self.weight_out[j][k]
# 시그모이드와 tanh 중에서 활성화 함수 선택
self.activation_out[k] = tanh(sum, False)

        return self.activation_out[:]

# 역전파의 실행. 각 층의 가중치가 변경된다.
# 출력층 은닉층 입력층 순서대로 변경한다.
def backPropagate(self, targets):

# 출력층 델타값 계산
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]

# 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
# 오차 * 활성화함수 미분 계수 값
            output_deltas[k] = tanh(self.activation_out[k], True) * error

# 은닉 노드의 오차 함수
        hidden_deltas = [0.0] * self.num_yh
        for j in range(self.num_yh):
            error = 0.0
            for k in range(self.num_yo):

# 출력층과는 오차 계산 방법이 다르므로 식이 달라진다. p.336 참고
                error = error + output_deltas[k] * self.weight_out[j][k]

# 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
# 오차 * 활성화함수 미분 계수 값
            hidden_deltas[j] = tanh(self.activation_hidden[j], True) * error

# 출력 가중치 업데이트
for j in range(self.num_yh):
            for k in range(self.num_yo):

# gradient가 오차를 가중치로 편미분한 것을 의미한다.
# activation_hidden이 y_h1이 된다.
                gradient = output_deltas[k] * self.activation_hidden[j]

# 모멘텀 SGD의 적용
                v = mo * self.gradient_out[j][k] - lr * gradient
                self.weight_out[j][k] += v
# 방금 움직였던 방향은 gradient를 그대로 기록
self.gradient_out[j][k] = gradient

# 입력 가중치 업데이트. 출력 가중치 업데이트와 같은 방식으로 적용
for i in range(self.num_x):
            for j in range(self.num_yh):
                gradient = hidden_deltas[j] * self.activation_input[i]
                v = mo*self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient

# 오차의 계산(최소 제곱법)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.activation_out[k]) ** 2
        return error

# 학습 실행
def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error: %-.5f' % error)

# 결괏값 출력
def result(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.update(p[0])))

# 실제 실행부
if __name__ == '__main__':

# 두 개의 입력 값, 두 개의 레이어, 하나의 출력 값을 갖도록 설정
    n = NeuralNetwork(2, 2, 1)

# 학습 실행
    n.train(data)

# 결괏값 출력
    n.result(data)


# Reference: http://arctrix.com/nas/python/bpnn.py (Neil Schemenauer)
