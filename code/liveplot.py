import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self):
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig = plt.figure(figsize=(12, 5))
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        self.x_arr = []
        self.loss_hist_train = []
        self.loss_hist_valid = []
        self.accuracy_hist_train = []
        self.accuracy_hist_valid = []

    def update_loss_train(self, epoch, loss_train): # 훈련 손실 그래프 업데이트
        if len(self.loss_hist_train) == 0 or self.x_arr[-1] != epoch: # 에포크 추가 조건문
            self.x_arr.append(epoch)  # 에포크 추가
            # 위 조건문을 넣지 않으면 x, y의 오류가 생김 -> x값인 에포크만 다른 함수들이 계속 추가하면 너무 커짐짐

        self.loss_hist_train.append(loss_train)
        self.plot_graphs()

    def update_loss_valid(self, epoch, loss_valid): # 검증 손실 그래프 업데이트
        if len(self.loss_hist_train) == 0 or self.x_arr[-1] != epoch:
            self.x_arr.append(epoch)  # 에포크 추가

        self.loss_hist_valid.append(loss_valid)
        self.plot_graphs()

    def update_accuracy_train(self, epoch, acc_train): # 훈련 정확도 그래프 업데이트
        if len(self.loss_hist_train) == 0 or self.x_arr[-1] != epoch:
            self.x_arr.append(epoch)  # 에포크 추가
            
        self.accuracy_hist_train.append(acc_train)
        self.plot_graphs()

    def update_accuracy_valid(self, epoch, acc_valid): # 검증 정확도 그래프 업데이트
        if len(self.loss_hist_train) == 0 or self.x_arr[-1] != epoch:
            self.x_arr.append(epoch)  # 에포크 추가
        
        self.accuracy_hist_valid.append(acc_valid)
        self.plot_graphs()

    def update_all(self, epoch, loss_train, loss_valid, acc_train, acc_valid): # 모든 그래프 업데이트
        if len(self.loss_hist_train) == 0 or self.x_arr[-1] != epoch:
            self.x_arr.append(epoch)  # 에포크 추가

        self.loss_hist_train.append(loss_train)
        self.loss_hist_valid.append(loss_valid)
        self.accuracy_hist_train.append(acc_train)
        self.accuracy_hist_valid.append(acc_valid)
        self.plot_graphs()

    def plot_graphs(self): # 그래프 그리는 함수
        self.ax1.clear()
        self.ax1.plot(self.x_arr[:len(self.loss_hist_train)], self.loss_hist_train, '-', label='Train loss')
        self.ax1.plot(self.x_arr[:len(self.loss_hist_valid)], self.loss_hist_valid, '--', label='Validation loss')
        self.ax1.set_xlabel('Epoch', size=15)
        self.ax1.set_ylabel('Loss', size=15)
        self.ax1.legend(fontsize=15)

        self.ax2.clear()
        self.ax2.plot(self.x_arr[:len(self.accuracy_hist_train)], self.accuracy_hist_train, '-', label='Train acc.')
        self.ax2.plot(self.x_arr[:len(self.accuracy_hist_valid)], self.accuracy_hist_valid, '--', label='Validation acc.')
        self.ax2.set_xlabel('Epoch', size=15)
        self.ax2.set_ylabel('Accuracy', size=15)
        self.ax2.legend(fontsize=15)

        plt.draw()
        plt.pause(0.1)

    def show(self): # 출력
        plt.show()
        plt.savefig("plot.png")