from AIGamePlatform import Othello
#from othello.bots.Random import BOT
from othello import OthelloGame
from othello.bots.DeepLearning import BOT
import time



class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))
        
BOARD_SIZE=8
bot=BOT(board_size=BOARD_SIZE)

args={
    'num_of_generate_data_for_train': 8,
    'epochs': 5,
    'batch_size': 4,
    'verbose': True
}
# bot.self_play_train(args)

app = Othello() # 會開啟瀏覽器登入Google Account，目前只接受@mail1.ncnu.edu.tw及@mail.ncnu.edu.tw

@app.competition(competition_id='test')
def _callback_(board, color): # 函數名稱可以自訂，board是當前盤面，color代表黑子或白子
    time.sleep(0.5)
    return bot.getAction(board, color) # 回傳要落子的座標

