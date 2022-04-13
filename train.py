from othello.bots.DeepLearning import BOT

BOARD_SIZE=8
bot=BOT(board_size = BOARD_SIZE)

args={
    'num_of_generate_data_for_train': 10,
    'epochs': 30,
    'batch_size': 8,
    'verbose': True
}

iterations = 9000

for _ in range(iterations):
    bot.self_play_train(args)


