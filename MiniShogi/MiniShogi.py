#
# MiniShogi.py 2020/9/18
#
RELEASE_CANDIDATE = True
TENSORFLOW = False
if TENSORFLOW:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import numpy as np
    MODELFILE = 'models/ms_model.h5'
if not RELEASE_CANDIDATE:
    import csv
    MOVESFILE = 'moves/ms_move.txt'
import random
import time
import pyxel
import msfont

END_WAIT   = 100
WRITE_MOVE =   0  # [20] 0:all
BOARD_X,    BOARD_Y    =  38,  18
TURNOVER_X, TURNOVER_Y =  12,   1
P1_STAND_X, P1_STAND_Y =   2,  18
P2_STAND_X, P2_STAND_Y = 136,  46
P1_X,       P1_Y       =   6,  88
P2_X,       P2_Y       = 140,  18
P1_X_PARAM, P1_Y_PARAM =   3, 117
P2_X_PARAM, P2_Y_PARAM = 137, 117
END_X,      END_Y      = 137, 116
MSG_X,      MSG_Y      =  38, 116
P1, P2, OLD_COL = 1, 2, 3
MSG_COL = {P1:7, P2:10, OLD_COL:12}
OPP     = {P1:P2, P2:P1}
VACANT, KING, LANCE, TOKIN, SILVER, BISHOP, GOLD, KNIGHT, ROOK, PAWN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
P1_AGGR, P2_AGGR = 10, 11
REV        = {-KING:-KING,   -LANCE:-TOKIN, -TOKIN:-LANCE, -SILVER:-BISHOP, -BISHOP:-SILVER, 
              -GOLD:-KNIGHT, -KNIGHT:-GOLD, -ROOK:-PAWN,   -PAWN:-ROOK,     -P1_AGGR:-P1_AGGR,
               KING: KING,    LANCE: TOKIN,  TOKIN: LANCE,  SILVER: BISHOP,  BISHOP: SILVER, 
               GOLD: KNIGHT,  KNIGHT: GOLD,  ROOK: PAWN,    PAWN: ROOK,      P2_AGGR: P2_AGGR}
PIECE_MOVE = {VACANT: [],                          # 0空き
              KING:   [ -6,-5,-4,-1, 1, 4, 5, 6],  # 1王
              LANCE:  [-10,],                      # 2香
              TOKIN:  [ -6,-5,-4,-1, 1, 5],        # 3と
              SILVER: [ -6,-5,-4, 4, 6],           # 4銀
              BISHOP: [-12,-8, 8,12],              # 5角
              GOLD:   [ -6,-5,-4,-1, 1, 5],        # 6金
              KNIGHT: [-11,-9],                    # 7桂
              ROOK:   [-10,-2, 2,10],              # 8飛
              PAWN:   [ -5,],                      # 9歩
              P1_AGGR:[],                          # 10:P1移動可能集約
              P2_AGGR:[]}                          # 11:P2移動可能集約
EIGHT_DIR  = {0:(1,5,6), 1:(-1,1,4,5,6), 2:(-1,1,4,5,6), 3:(-1,1,4,5,6), 4:(-1,4,5), 
              5:(-5,-4,1,5,6), 10:(-5,-4,1,5,6), 15:(-5,-4,1,5,6), 
              9:(-6,-5,-1,4,5), 14:(-6,-5,-1,4,5), 19:(-6,-5,-1,4,5), 
              20:(-5,-4,1), 21:(-6,-5,-4,-1,1), 22:(-6,-5,-4,-1,1), 23:(-6,-5,-4,-1,1), 24:(-6,-5,-1)}
DEF_8_DIR  = (-6,-5,-4,-1,1,4,5,6)
EDGE1      = (0, 5, 10, 15, 20)
EDGE2      = (4, 9, 14, 19, 24)
BACKRANK1  = (0, 1, 2, 3, 4)
BACKRANK2  = (20, 21, 22, 23, 24)
STATUS_TITLE               = 110
STATUS_START               = 120
STATUS_COUNT_CHECKED       = 130
STATUS_COM_S               = 200
STATUS_COM_INIT_CHECKMATE  = 210
STATUS_COM_COUNT_CHECKMATE = 220
STATUS_COM_INIT_BRINKMATE  = 230
STATUS_COM_COUNT_BRINKMATE = 240
STATUS_COM_METHOD          = 250
STATUS_COM_RULE            = 260
STATUS_COM_AI              = 270
STATUS_COM_CHOOSE          = 280
STATUS_COM_E               = 299
STATUS_MAN_SELECT          = 310
STATUS_MAN_DROP            = 320
STATUS_CHECKMATE_FLUSH     = 410
STATUS_MESSAGE             = 420
STATUS_END                 = 510
CHARA_NAME = ('ピノキオ', 'ピーターパン', '人魚', '桃太郎', '金太郎', '裸の王様', 
              'ホームズ', '赤ちゃん', '孫悟空', '三蔵法師', 'エイリアン', '忍者')
CHARA_PER  = ('真面目な', '豪快な', '意地悪な', '気まぐれな', '臆病な', 'ずる賢い', 
              '粘り強い', '気が荒い', '欲張りな', '能天気な', '慎重な', '頑固な')
RULE, RULE_MODEL, MODEL_RULE, MODEL, MODEL3, MODEL1 = 0, 1, 2, 3, 4, 5
if TENSORFLOW:
    CHARA_METHOD = {0:RULE, 1:RULE_MODEL, 2:MODEL_RULE, 3:MODEL,  4:MODEL3,  5:MODEL1, 
                    6:RULE, 7:RULE_MODEL, 8:MODEL_RULE, 9:MODEL, 10:MODEL3, 11:MODEL1}
else:
    CHARA_METHOD = {0:RULE, 1:RULE, 2:RULE, 3:RULE,  4:RULE,  5:RULE, 
                    6:RULE, 7:RULE, 8:RULE, 9:RULE, 10:RULE, 11:RULE}
CHARA_MSG = {RULE:'Rule', RULE_MODEL:'Rule\n to\n  Model', MODEL_RULE:'Model\n to\n  Rule', 
             MODEL:'Model\n All\n  Moves', MODEL3:'Model\n Best3', MODEL1:'Model\n Best1'} 

class Message:
    def __init__(self, x, y, width, line, frcol=7, bgcol=0, height=0):
        self.msg_x = x
        self.msg_y = y
        self.msg_width = width
        self.msg_line = line
        self.msg_frcol = frcol
        self.msg_bgcol = bgcol
        if height < line*8+3:
            self.msg_height = line*8+3
        else:
            self.msg_height = height
        self.msg_scrl = 0
        self.msg_col = 7
        self.msg_old_col = MSG_COL[OLD_COL]
        self.clr()
    
    def clr(self):
        self.msg_str = ['']*self.msg_line

    def in_message(self, new_msg, turn=P1, keep=False):
        if turn==P2:
            new_msg = ' '*(11-len(new_msg))*2+' '+new_msg
        self.msg_col = MSG_COL[turn]
        if keep or self.msg_str[0]=='':
            self.msg_str[0] = new_msg
        elif new_msg:
            for i in reversed(range(self.msg_line-1)):
                self.msg_str[i+1] = self.msg_str[i]
            self.msg_str[0] = new_msg
            self.msg_scrl = 8

    def draw_message(self):
        pyxel.rectb(self.msg_x, self.msg_y, self.msg_width, self.msg_height, self.msg_frcol)
        pyxel.rect(self.msg_x+1, self.msg_y+1, self.msg_width-2, self.msg_height-2, self.msg_bgcol)
        for i in range(1, self.msg_line):
            msfont.text(self.msg_x+2, self.msg_y+2+(self.msg_line-i-1)*8+self.msg_scrl, self.msg_str[i], self.msg_old_col)
        if self.msg_scrl==0:
            msfont.text(self.msg_x+2, self.msg_y+2+(self.msg_line-1)*8, self.msg_str[0], self.msg_col)

    def scroll(self):
        if self.msg_scrl > 0:
            self.msg_scrl -= 1
            return True
        return False

class App:
    def reset_start(self):
        self.board = [-PAWN,-GOLD,-KING,-SILVER,-TOKIN, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, TOKIN,SILVER,KING,GOLD,PAWN]
        self.inhand_p1 = []
        self.inhand_p2 = []
        self.allmove = []
        self.spc1 = self.spc2 = 0
        self.msg1 = self.msg2 = ''
        self.msg_scrl = 0
        self.col1 = self.col2 = MSG_COL[P1]
        self.turn = random.choice((P1, P2))
        self.select_pos = -1
        self.drop_piece = 0
        self.selected_pos = -1
        self.dropped_pos  = -1
        self.prev_move_num = 0
        self.capture_piece = []
        self.shake_p1 = self.shake_p2 = 0
        self.is_continue = self.is_stop = self.is_brinkmate = False
        self.move_hist = []
        self.dsp_param  = {P1:CHARA_MSG[CHARA_METHOD[self.chara[P1]]], P2:CHARA_MSG[CHARA_METHOD[self.chara[P2]]]}
        self.dsp_select = {P1:0 , P2:0 }
        self.com_move = []
        self.checkcont = {P1:0, P2:0}
        self.msg.in_message(CHARA_PER[self.chara[P1]]+'「'+CHARA_NAME[self.chara[P1]]+'」', P1)
        self.msg.in_message(CHARA_PER[self.chara[P2]]+'「'+CHARA_NAME[self.chara[P2]]+'」', P2)
    
    def __init__(self):
        pyxel.init(172, 137, title='MiniShogi')
        pyxel.load('assets/MiniShogi.pyxres')
        pyxel.mouse(True)
        if TENSORFLOW:
            self.ms_model = tf.keras.models.load_model(MODELFILE)
        self.is_man_man = self.is_man_com = self.is_com_com = False
        self.is_ccstrict = True
        self.chara = {P1:4, P2:11}
        self.msg = Message(MSG_X, MSG_Y, 96, 2)
        self.reset_start()
        self.status = STATUS_TITLE
        pyxel.run(self.update, self.draw)
    
    def append_canmove_bd(self, turn, bd, ih_p1, ih_p2, src_pos, diff):
        ret = []
        if diff in (-12, -10, -8, -2, 2, 8, 10, 12):
            diff //= 2
            lp = True
        else:
            lp = False
        dst_pos = src_pos + diff
        while True:
            cp_bd = bd[:]
            cp_ih_p1 = ih_p1[:]
            cp_ih_p2 = ih_p2[:]
            if dst_pos < 0 or 25 <= dst_pos:
                break
            dst_piece = bd[dst_pos]
            if (turn == P1 and dst_piece < 0) or (turn == P2 and dst_piece > 0) or \
                    (dst_pos in (0,5,10,15,20)) and (diff in (-9,-4,1,6,11)) or \
                    (dst_pos in (4,9,14,19,24)) and (diff in (-11,-6,-1,4,9)):
                break
            if dst_piece:
                if turn == P1:
                    cp_ih_p1.append(-dst_piece)
                else:
                    cp_ih_p2.append(-dst_piece)
            cp_bd[dst_pos] = REV[bd[src_pos]]
            cp_bd[src_pos] = VACANT
            ret.append([cp_bd, cp_ih_p1, cp_ih_p2, src_pos, dst_pos])
            if (turn == P1 and dst_piece > 0) or (turn == P2 and dst_piece < 0) or not lp:
                break
            dst_pos += diff
        return ret
    
    def append_canmove_std(self, turn, bd, ih_p1, ih_p2, src_pos, dst_pos):
        ret = []
        cp_bd1   = bd[:]
        cp_ih_p1 = ih_p1[:]
        cp_ih_p2 = ih_p2[:]
        if turn == P1:
            move_piece = cp_ih_p1.pop(src_pos)
            cp_bd1[dst_pos] = move_piece
            if not ((move_piece == -PAWN and 20 <= dst_pos < 25) or \
                    (move_piece == -LANCE and 20 <= dst_pos < 25) or \
                    (move_piece == -KNIGHT and 15 <= dst_pos < 25)):  # 行き所のない打つ手を除く
                ret.append([cp_bd1, cp_ih_p1, cp_ih_p2, src_pos+100, dst_pos])
            if cp_bd1[dst_pos] != -P1_AGGR:
                cp_bd2 = cp_bd1[:]
                move_piece = REV[move_piece]
                cp_bd2[dst_pos] = move_piece
                if not ((move_piece == -PAWN and 20 <= dst_pos < 25) or \
                        (move_piece == -LANCE and 20 <= dst_pos < 25) or \
                        (move_piece == -KNIGHT and 15 <= dst_pos < 25)):  # 行き所のない打つ手を除く
                    ret.append([cp_bd2, cp_ih_p1, cp_ih_p2, src_pos+100, dst_pos])
        else:
            move_piece = cp_ih_p2.pop(src_pos)
            cp_bd1[dst_pos] = move_piece
            if not ((move_piece == PAWN and 0 <= dst_pos < 5) or \
                    (move_piece == LANCE and 0 <= dst_pos < 5) or \
                    (move_piece == KNIGHT and 0 <= dst_pos < 10)):  # 行き所のない打つ手を除く
                ret.append([cp_bd1, cp_ih_p1, cp_ih_p2, src_pos+200, dst_pos])
            if cp_bd1[dst_pos] != P2_AGGR:
                cp_bd2 = cp_bd1[:]
                move_piece = REV[move_piece]
                cp_bd2[dst_pos] = move_piece
                if not ((move_piece == PAWN and 0 <= dst_pos < 5) or \
                        (move_piece == LANCE and 0 <= dst_pos < 5) or \
                        (move_piece == KNIGHT and 0 <= dst_pos < 10)):  # 行き所のない打つ手を除く
                    ret.append([cp_bd2, cp_ih_p1, cp_ih_p2, src_pos+200, dst_pos])
        return ret
    
    def canmove(self, turn, bd, ih_p1, ih_p2):
        ret = []
        for src_pos in range(25):
            if (turn == P1 and bd[src_pos] < 0) or (turn == P2 and bd[src_pos] > 0):
                for diff in PIECE_MOVE[abs(bd[src_pos])]:
                    if turn == P1:
                        diff = -diff
                    ret.extend(self.append_canmove_bd(turn, bd, ih_p1, ih_p2, src_pos, diff))
        if turn == P1:
            for src_pos in range(len(ih_p1)):
                for dst_pos in range(25):
                    if bd[dst_pos] == VACANT:
                        ret.extend(self.append_canmove_std(turn, bd, ih_p1, ih_p2, src_pos, dst_pos))
        else:
            for src_pos in range(len(ih_p2)):
                for dst_pos in range(25):
                    if bd[dst_pos] == VACANT:
                        ret.extend(self.append_canmove_std(turn, bd, ih_p1, ih_p2, src_pos, dst_pos))
        return ret
    
    def count_overlap(self, turn, bd):
        ret1 = [VACANT]*25
        ret2 = [VACANT]*25
        for src_pos, src_piece in enumerate(bd):
            for diff in PIECE_MOVE[abs(src_piece)]:
                if diff in (-12, -10, -8, -2, 2, 8, 10, 12):
                    diff //= 2
                    lp = True
                else:
                    lp = False
                if src_piece < 0:  # P1
                    diff = -diff
                dst_pos = src_pos + diff
                while True:
                    if (dst_pos < 0 or 25 <= dst_pos) or \
                            (dst_pos in (0,5,10,15,20)) and (diff in (-9,-4,1,6,11)) or \
                            (dst_pos in (4,9,14,19,24)) and (diff in (-11,-6,-1,4,9)):
                        break
                    ret1[dst_pos] += 1 if (turn == P1 and src_piece < 0) or (turn == P2 and src_piece > 0) else -1
                    ret2[dst_pos] += 1
                    if not lp:
                        break
                    dst_pos += diff
        return ret1, ret2
    
    def form_bd_ih(self, src_move):
        ret_bd = src_move[0][:]
        ih_p1  = src_move[1][:]
        ih_p2  = src_move[2][:]
        for i in range(len(ih_p1)):  # 裏反転そろえる
            if ih_p1[i] < REV[ih_p1[i]]:
                ih_p1[i] = REV[ih_p1[i]]
        ih_p1.sort(reverse=True)
        ih_p1.extend([0]*(8-len(ih_p1)))  # 0-padding
        ret_bd.extend(ih_p1)
        for i in range(len(ih_p2)):  # 裏反転そろえる
            if ih_p2[i] > REV[ih_p2[i]]:
                ih_p2[i] = REV[ih_p2[i]]
        ih_p2.sort()
        ih_p2.extend([0]*(8-len(ih_p2)))  # 0-padding
        ret_bd.extend(ih_p2)
        return ret_bd
    
    def append_move(self, src_move):
        bd = self.form_bd_ih(src_move)
        repmove = self.move_hist.count(bd)
        if repmove == 2:
            self.msg.in_message('（同一局面３回）', self.turn)
        elif repmove > 2:
            self.msg.in_message('千日手で勝負なし', self.turn)
            self.turn = OPP[self.turn]
            return True
        self.move_hist.append(bd)
        return False
    
    def flip(self, src_move):
        ret_move = [VACANT]*41
        for i in range(25):
            ret_move[i] = -src_move[24-i]
        for i in range(8):
            ret_move[25  +i] = -src_move[25+8+i]
            ret_move[25+8+i] = -src_move[25  +i]
        return ret_move
    
    def flush_move(self, win):
        write_move_hist = []
        for i, one_move in enumerate(reversed(self.move_hist[-WRITE_MOVE:])):
            # (i偶数==win/win==P2 => P2:反転) or (i奇数==lose/win==P1 => P2:反転)
            if (i%2 == 0 and win == P2) or (i%2 == 1 and win == P1):
                one_move = self.flip(one_move)
            #one_move.extend([i])  # 最後からの手数
            one_move.extend([1-i%2, i%2])  # win->[1,0] defeat->[0,1]
            write_move_hist.append(one_move)
        with open(MOVESFILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(write_move_hist)
    
    def update(self):
        t = time.perf_counter()
        for _ in range(16):
            self.update2()
            if time.perf_counter() > t+0.03 or self.status in (STATUS_TITLE, STATUS_MAN_SELECT, STATUS_MAN_DROP, STATUS_END):
                break

    def update2(self):
        if self.msg.scroll():
            return
        if self.shake_p1:
            self.shake_p1 -= 1
        if self.shake_p2:
            self.shake_p2 -= 1
        
        if self.is_com_com and pyxel.btnr(pyxel.MOUSE_BUTTON_RIGHT):  # RIGHT_BUTTON_UP/機械学習中断
            self.reset_start()
            self.status = STATUS_TITLE
        
        if self.status == STATUS_TITLE:
            self.is_man_man = (BOARD_X+4<pyxel.mouse_x<BOARD_X+89 and BOARD_Y+30<pyxel.mouse_y<BOARD_Y+40)
            self.is_man_com = (BOARD_X+4<pyxel.mouse_x<BOARD_X+89 and BOARD_Y+42<pyxel.mouse_y<BOARD_Y+52)
            self.is_com_com = (BOARD_X+4<pyxel.mouse_x<BOARD_X+89 and BOARD_Y+54<pyxel.mouse_y<BOARD_Y+64)
            self.is_ccstrict = not pyxel.btn(pyxel.KEY_SHIFT)
            if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):  # LEFT_BUTTON_UP
                if P1_X<pyxel.mouse_x<P1_X+25 and P1_Y<pyxel.mouse_y<P1_Y+25:
                    self.chara[P1] += 1
                    if self.chara[P1] > 5:
                        self.chara[P1] = 0
                    self.msg.in_message(CHARA_PER[self.chara[P1]]+'「'+CHARA_NAME[self.chara[P1]]+'」', P1)
                    self.dsp_param[P1] = CHARA_MSG[CHARA_METHOD[self.chara[P1]]]
                elif P2_X<pyxel.mouse_x<P2_X+25 and P2_Y<pyxel.mouse_y<P2_Y+25:
                    self.chara[P2] += 1
                    if self.chara[P2] > 11:
                        self.chara[P2] = 6
                    self.msg.in_message(CHARA_PER[self.chara[P2]]+'「'+CHARA_NAME[self.chara[P2]]+'」', P2)
                    self.dsp_param[P2] = CHARA_MSG[CHARA_METHOD[self.chara[P2]]]
                elif self.is_man_man or self.is_man_com or self.is_com_com:
                    self.status = STATUS_START
        
        elif self.status == STATUS_START:  # 開始/全ての手
            self.bd_overlap, self.bd_focus = self.count_overlap(self.turn, self.board)
            self.allmove = self.canmove(self.turn, self.board, self.inhand_p1, self.inhand_p2)  # 自分の手
            for i in reversed(range(len(self.allmove))):  # 王が取られる手を除く
                each1 = self.allmove[i]
                for each2 in self.canmove(OPP[self.turn], each1[0], each1[1], each1[2]):
                    if (self.turn == P1 and KING in each2[2]) or (self.turn == P2 and -KING in each2[1]):
                        del self.allmove[i]
                        break
            self.status = STATUS_COUNT_CHECKED
        
        elif self.status == STATUS_COUNT_CHECKED:  # 王手されているか
            for each1 in self.canmove(OPP[self.turn], self.board, self.inhand_p1, self.inhand_p2):  # 相手の手
                if -KING in each1[1]:  # self.turn == P2
                    self.shake_p2 = 8
                    break
                elif KING in each1[2]:  # self.turn == P1 
                    self.shake_p1 = 8
                    break
            if self.is_com_com or (self.is_man_com and self.turn == P1):
                self.status = STATUS_COM_INIT_CHECKMATE
            else:
                self.status = STATUS_MAN_SELECT
        
        elif self.status == STATUS_COM_INIT_CHECKMATE:  # 詰み確認準備
            self.lp1 = 0
            self.checkmate = []
            self.status = STATUS_COM_COUNT_CHECKMATE
        
        elif self.status == STATUS_COM_COUNT_CHECKMATE:  # 詰み確認
            each1 = self.allmove[self.lp1]
            self.oppmove = self.canmove(OPP[self.turn], each1[0], each1[1], each1[2])  # 次(相手)
            checkmate_num = 0
            for each2 in self.oppmove:
                if len(each2[1]) >= 2:
                    PIECE_MOVE[P1_AGGR] = list(set([y for x in each2[1] \
                            for y in PIECE_MOVE[abs(x)]+PIECE_MOVE[abs(REV[x])]]))  # P1
                    each2[1] = [-P1_AGGR]
                if len(each2[2]) >= 2:
                    PIECE_MOVE[P2_AGGR] = list(set([y for x in each2[2] \
                            for y in PIECE_MOVE[x]+PIECE_MOVE[REV[x]]]))  # P2
                    each2[2] = [ P2_AGGR]
                ownmove_ = self.canmove(self.turn, each2[0], each2[1], each2[2])  # 次の次(自分)
                for each3 in ownmove_:
                    if -KING in each3[1] or KING in each3[2]:
                        checkmate_num += 1
                        break
            if checkmate_num == len(self.oppmove):
                self.checkmate.append(self.lp1)
            self.lp1 += 1
            if self.lp1 >= len(self.allmove):
                if self.checkmate:
                    self.dsp_param[self.turn] = f'Check-\n    mate\n{len(self.checkmate)}move(s)'
                    self.com_move = self.allmove[random.choice(self.checkmate)]
                    self.status = STATUS_COM_CHOOSE
                else:
                    if self.is_com_com or (self.is_man_com and self.turn == P1):
                        self.status = STATUS_COM_INIT_BRINKMATE if self.is_ccstrict else STATUS_COM_METHOD
        
        elif self.status == STATUS_COM_INIT_BRINKMATE:  # 必至確認準備
            self.lp1 = 0
            self.lp2 = 0
            self.brinkmate = []
            self.is_brinkmate = False
            self.msg.in_message('（考え中　　　）', self.turn)
            self.status = STATUS_COM_COUNT_BRINKMATE
        
        elif self.status == STATUS_COM_COUNT_BRINKMATE:  # 必至確認
            if pyxel.btnr(pyxel.MOUSE_BUTTON_RIGHT):  # RIGHT_BUTTON_UP
                self.lp1 = 0
                self.lp2 = 0
                self.msg.in_message('（中断しました）', self.turn, True)
                self.status = STATUS_COM_METHOD
            else:
                if self.lp2 == 0:
                    self.msg.in_message('（考え中'+'．'*(self.lp1%4)+'　'*(3-(self.lp1%4))+'）', self.turn, True)
                    each1 = self.allmove[self.lp1]  # (自分)
                    self.oppmove = self.canmove(OPP[self.turn], each1[0], each1[1], each1[2])  # 次(相手)
                each2 = self.oppmove[self.lp2]
                if len(each2[1]) >= 2:
                    PIECE_MOVE[P1_AGGR] = list(set([y for x in each2[1] \
                            for y in PIECE_MOVE[abs(x)]+PIECE_MOVE[abs(REV[x])]]))  # P1
                    each2[1] = [-P1_AGGR]
                if len(each2[2]) >= 2:
                    PIECE_MOVE[P2_AGGR] = list(set([y for x in each2[2] \
                            for y in PIECE_MOVE[x]+PIECE_MOVE[REV[x]]]))  # P2
                    each2[2] = [ P2_AGGR]
                ownmove = self.canmove(self.turn, each2[0], each2[1], each2[2])  # 次の次(自分)
                brinkmate_num = 0
                for each3 in ownmove:
                    if -KING in each3[1] or KING in each3[2]:
                        break
                    opp2move = self.canmove(OPP[self.turn], each3[0], each3[1], each3[2])  # 次の次の次(相手)
                    for each4 in opp2move:
                        if -KING in each4[1] or KING in each4[2]:
                            brinkmate_num += 1
                            break
                if brinkmate_num == len(ownmove):
                    self.brinkmate.append(self.lp1)
                    self.lp2 = len(self.oppmove)
                self.lp2 += 1
                if self.lp2 >= len(self.oppmove):
                    self.lp2 = 0
                    self.lp1 += 1
                    if self.lp1 >= len(self.allmove):
                        self.msg.in_message('', self.turn, True)  # 考え中を消す
                        if len(self.brinkmate) == len(self.allmove):  # 必至
                            self.dsp_param[self.turn] = f'Brink-\n    mate\n{len(self.brinkmate)}move(s)'
                            self.is_brinkmate = True
                        else:
                            for i in reversed(self.brinkmate):  # 必至を削除
                                del self.allmove[i]
                        self.status = STATUS_COM_METHOD
        
        elif self.status == STATUS_COM_METHOD:
            if CHARA_METHOD[self.chara[self.turn]] in (RULE, RULE_MODEL):
                self.status = STATUS_COM_RULE
            else:
                self.status = STATUS_COM_AI
        
        elif self.status == STATUS_COM_RULE:  # 考える(ルール)
            focus1       = []
            focus2       = []
            notcaptured  = []
            owncapture   = []
            owncheck     = []
            decoppking   = []
            dec_captured = []
            ownih_num    = len(self.inhand_p1 if self.turn == P1 else self.inhand_p2)  # 持ち駒数（自分）
            oppih_num    = len(self.inhand_p2 if self.turn == P1 else self.inhand_p1)  # 持ち駒数（相手）
            oppking_num  = 0
            captured_num = 0
            oppking_pos  = self.board.index(KING if self.turn == P1 else -KING)
            for each1 in self.canmove(OPP[self.turn], self.board, self.inhand_p1, self.inhand_p2):  # (相手)
                if oppking_pos == each1[3]:
                    for each2 in self.canmove(self.turn, each1[0], each1[1], each1[2]):
                        if -KING in each2[1] or KING in each2[2]:
                            break
                    else:
                        oppking_num += 1  # 相手の玉が動ける手の数
                if len(each1[2] if self.turn == P1 else each1[1]) > oppih_num:
                    captured_num += 1  # 取られる手の数
            for i, each1 in enumerate(self.allmove):  # (自分)
                if self.bd_overlap[each1[4]] >= 1 or (self.bd_overlap[each1[4]] >= 0 and each1[3] >= 100):
                    focus1.append(i)  # 焦点1
                if self.bd_overlap[each1[4]] >= 2 or (self.bd_overlap[each1[4]] >= 1 and each1[3] >= 100):
                    focus2.append(i)  # 焦点2
                if len(each1[self.turn]) > ownih_num:
                    owncapture.append(i)  # 取れる手
                oppmove = self.canmove(OPP[self.turn], each1[0], each1[1], each1[2])
                for each2 in oppmove:  # 次(相手)
                    if len(each2[OPP[self.turn]]) > oppih_num:
                        break
                else:
                    notcaptured.append(i)  # 一つも取られない
                each_captured_num = 0
                each_oppking_num = 0
                each_oppking_pos = each1[0].index(KING if self.turn == P1 else -KING)
                for each2 in oppmove:  # 次(相手)
                    if len(each2[2] if self.turn == P1 else each2[1]) > oppih_num:
                        each_captured_num += 1
                    if each_oppking_pos == each2[3]:
                        for each3 in self.canmove(self.turn, each2[0], each2[1], each2[2]):  # 次の次(自分)
                            if -KING in each3[1] or KING in each3[2]:
                                break
                        else:
                            each_oppking_num += 1
                if each_captured_num < captured_num:
                    dec_captured.append(i)  # 取られる駒数が減る
                if each_oppking_num < oppking_num:
                    decoppking.append(i)  # 相手の玉が動ける手が減る手
                ownmove = self.canmove(self.turn, each1[0], each1[1], each1[2])  # 次(自分)
                for each2 in ownmove:
                    if -KING in each2[1] or KING in each2[2]:
                        owncheck.append(i)  # 王手
                        break
            check_nocpt_fcs    = list(set(owncheck) & (set(notcaptured) | set(focus1)))
            capt_check_notcapt = list(set(owncapture) & (set(owncheck) | set(notcaptured)))
            decking_nocpt_fcs  = list(set(decoppking) & (set(notcaptured) | set(focus1)))
            other              = list(set(owncapture) | set(focus2) | set(decoppking))
            if check_nocpt_fcs:  # 王手(次に取られない/焦点が多い)
                self.dsp_param[self.turn] = f'Check\n{len(check_nocpt_fcs)}move(s)'  # 'Check\nNoCapFcs'
                self.com_move = self.allmove[random.choice(check_nocpt_fcs)]
            elif capt_check_notcapt:  # 取る(王手/次に取られない)
                self.dsp_param[self.turn] = f'Capture\n{len(capt_check_notcapt)}move(s)'  # 'Capture\nChkNoCap'
                self.com_move = self.allmove[random.choice(capt_check_notcapt)]
            elif decking_nocpt_fcs:  # 相手の玉が動ける手が減る手(次に取られない/焦点が多い)
                self.dsp_param[self.turn] = f'Movable\n{len(decking_nocpt_fcs)}move(s)'  # 'DecKngMv\nNoCapFcs'
                self.com_move = self.allmove[random.choice(decking_nocpt_fcs)]
            elif len(self.allmove) >= 2 and CHARA_METHOD[self.chara[self.turn]] == RULE_MODEL:
                self.status = STATUS_COM_AI
                return
            elif dec_captured:  # 取られる駒数が減る
                self.dsp_param[self.turn] = f'Escape\n{len(dec_captured)}move(s)'  # 'DecCaptd'
                self.com_move = self.allmove[random.choice(dec_captured)]
            elif other:  # その他
                self.dsp_param[self.turn] = f'Other\n{len(other)}move(s)'
                self.com_move = self.allmove[random.choice(other)]
            else:  # ランダム
                self.dsp_param[self.turn] = f'Random\n{len(self.allmove)}move(s)'
                self.com_move = random.choice(self.allmove)
            self.status = STATUS_COM_CHOOSE
        
        elif self.status == STATUS_COM_AI:  # 考える(モデル)
            bd_all = []
            for each in self.allmove:
                bd = self.form_bd_ih(each)
                if self.turn == P2:
                    bd = self.flip(bd)
                bd_all.append(bd)
            prob = self.ms_model.predict_on_batch(np.array(bd_all))
            eval_move = []
            for p in prob:
                eval_move.append(p[0])
            if CHARA_METHOD[self.chara[self.turn]] == MODEL1:
                selected = eval_move.index(max(eval_move))  # 1位
                self.com_move = self.allmove[selected]
            else:  # MODEL3, MODEL
                if CHARA_METHOD[self.chara[self.turn]] == MODEL3:
                    eval_best3 = sorted(eval_move, reverse=True)[min(len(eval_move)-1, 2)]  # 3位
                    eval_move = [0 if e<eval_best3 else e for e in eval_move]  # 4位以降0.0
                eval_move = [e/max(eval_move) for e in eval_move]  # 1位1.0
                eval_move = [e**7 for e in eval_move]  # 全て7乗
                selected, self.com_move = random.choices(list(enumerate(self.allmove)), weights=eval_move)[0]
                if len(self.allmove) >= 2 and eval_move[selected] < 0.5 and CHARA_METHOD[self.chara[self.turn]] == MODEL_RULE:
                    self.status = STATUS_COM_RULE
                    return
            self.dsp_param[self.turn] = sorted(eval_move, reverse=True)
            self.dsp_select[self.turn] = eval_move[selected]
            self.status = STATUS_COM_CHOOSE
        
        elif self.status == STATUS_COM_CHOOSE:  # 手を決める
            if self.turn == P1:
                self.capture_piece = list(set(self.com_move[1])-set(self.inhand_p1))
            else:
                self.capture_piece = list(set(self.com_move[2])-set(self.inhand_p2))
            self.prev_move_num = len(self.allmove)
            self.board         = self.com_move[0]
            self.inhand_p1     = self.com_move[1]
            self.inhand_p2     = self.com_move[2]
            self.selected_pos  = self.com_move[3]
            self.dropped_pos   = self.com_move[4]
            if(self.append_move(self.com_move)):  # 千日手
                self.end_wait = 0
                self.status = STATUS_END
                return
            self.status = STATUS_CHECKMATE_FLUSH
        
        elif self.status == STATUS_MAN_SELECT:
            if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):  # LEFT_BUTTON_UP
                select_x = (pyxel.mouse_x-BOARD_X)//19
                select_y = (pyxel.mouse_y-BOARD_Y)//19
                if 0 <= select_x < 5 and 0 <= select_y < 5:
                    self.select_pos = select_y*5+select_x
                    if self.select_pos in [v[3] for v in self.allmove]:
                        self.drop_piece = REV[self.board[self.select_pos]]
                        self.status = STATUS_MAN_DROP
                else:
                    if self.turn == P1:
                        select_x = (pyxel.mouse_x-P1_STAND_X-2)//15
                        select_y = (pyxel.mouse_y-P1_STAND_Y-2)//16
                        if 0 <= select_x < 2 and 0 <= select_y < 4:
                            n = 6-select_y*2+1-select_x
                            self.select_pos = 100+n
                            if self.select_pos in [v[3] for v in self.allmove]:
                                self.drop_piece = self.inhand_p1[n]
                                self.status = STATUS_MAN_DROP
                    else:
                        select_x = (pyxel.mouse_x-P2_STAND_X-2)//15
                        select_y = (pyxel.mouse_y-P2_STAND_Y-2)//16
                        if 0 <= select_x < 2 and 0 <= select_y < 4:
                            n = select_y*2+select_x
                            self.select_pos = 200+n
                            if self.select_pos in [v[3] for v in self.allmove]:
                                self.drop_piece = self.inhand_p2[n]
                                self.status = STATUS_MAN_DROP
        
        elif self.status == STATUS_MAN_DROP:
            if pyxel.btnr(pyxel.MOUSE_BUTTON_RIGHT):  # RIGHT_BUTTON_UP
                self.status = STATUS_MAN_SELECT
            elif pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):  # LEFT_BUTTON_UP
                select_x = (pyxel.mouse_x-BOARD_X)//19
                select_y = (pyxel.mouse_y-BOARD_Y)//19
                new_drop_pos = select_y*5+select_x
                if 0 <= select_x < 5 and 0 <= select_y < 5:
                    for i in range(len(self.allmove)):
                        if self.select_pos == self.allmove[i][3] and new_drop_pos == self.allmove[i][4] and \
                                self.drop_piece == self.allmove[i][0][new_drop_pos]:
                            if self.turn == P1:
                                self.capture_piece = list(set(self.allmove[i][1])-set(self.inhand_p1))
                            else:
                                self.capture_piece = list(set(self.allmove[i][2])-set(self.inhand_p2))
                            self.prev_move_num = len(self.allmove)
                            self.board         = self.allmove[i][0]
                            self.inhand_p1     = self.allmove[i][1]
                            self.inhand_p2     = self.allmove[i][2]
                            self.selected_pos  = self.allmove[i][3]
                            self.dropped_pos   = self.allmove[i][4]
                            if(self.append_move(self.allmove[i])):  # 千日手
                                self.status = STATUS_END
                                return
                            self.status = STATUS_CHECKMATE_FLUSH
                            break
                    else:
                        self.status = STATUS_MAN_SELECT
                else:
                    if self.turn == P1:
                        select_x = (pyxel.mouse_x-P1_STAND_X-2)//15
                        select_y = (pyxel.mouse_y-P1_STAND_Y-2)//16
                        n = 6-select_y*2+1-select_x
                        if 0 <= select_x < 2 and 0 <= select_y < 4 and self.select_pos == n+100:  # 持ち駒反転
                            self.inhand_p1[n] = REV[self.inhand_p1[n]]
                            self.drop_piece = self.inhand_p1[n]
                        else:
                            self.status = STATUS_MAN_SELECT
                    else:
                        select_x = (pyxel.mouse_x-P2_STAND_X-2)//15
                        select_y = (pyxel.mouse_y-P2_STAND_Y-2)//16
                        n = select_y*2+select_x
                        if 0 <= select_x < 2 and 0 <= select_y < 4 and self.select_pos == n+200:  # 持ち駒反転
                            self.inhand_p2[n] = REV[self.inhand_p2[n]]
                            self.drop_piece = self.inhand_p2[n]
                        else:
                            self.status = STATUS_MAN_SELECT
        
        elif self.status == STATUS_CHECKMATE_FLUSH:
            checkmate_num = 0
            oppallmove = self.canmove(OPP[self.turn], self.board, self.inhand_p1, self.inhand_p2)  # 次（相手）
            for each1 in oppallmove:  # 王が取られる手を除く
                for each2 in self.canmove(self.turn, each1[0], each1[1], each1[2]):  # 次の次（自分）
                    if -KING in each2[1] or KING in each2[2]:
                        checkmate_num += 1
                        break
                else:
                    break
            self.ischeckmate = (checkmate_num == len(oppallmove))
            if self.ischeckmate and not RELEASE_CANDIDATE:
                self.flush_move(self.turn)
            self.status = STATUS_MESSAGE
        
        elif self.status == STATUS_MESSAGE:
            ownking_pos = self.board.index(-KING if self.turn == P1 else  KING)
            oppking_pos = self.board.index( KING if self.turn == P1 else -KING)
            moved_piece = self.board[self.dropped_pos]
            captured = False  # ぶつける（指した駒が次に取られる）
            for each1 in self.canmove(OPP[self.turn], self.board, self.inhand_p1, self.inhand_p2):
                if self.dropped_pos == each1[4]:
                    captured = True
                    break
            doublecheck_num = 0  # 王手している駒数（両王手）
            capture_num = 0      # 指した駒で次に取れる駒数
            discovd_chk = False  # 開き王手
            forkmove    = False  # 両取り
            for each1 in self.canmove(self.turn, self.board, self.inhand_p1, self.inhand_p2):
                if (self.turn == P1 and -KING in each1[1]) or (self.turn == P2 and KING in each1[2]):
                    doublecheck_num += 1
                    if REV[moved_piece] != each1[0][each1[4]]:
                        discovd_chk = True
                if self.dropped_pos == each1[3] and \
                        ((self.turn == P1 and len(each1[1]) > len(self.inhand_p1)) or \
                         (self.turn == P2 and len(each1[2]) > len(self.inhand_p2))):
                    capture_num += 1
            if doublecheck_num:
                self.checkcont[self.turn] += 1
            else:
                self.checkcont[self.turn] = 0
            if capture_num >= 2 and not captured and not abs(moved_piece) == KING:
                forkmove = True
            knightfork = False  # ふんどしの桂
            if forkmove and abs(moved_piece) == KNIGHT:
                knightfork = True
            silverfork = False  # 割り銀
            if forkmove and abs(moved_piece) == SILVER:
                silverfork = True
            bishopfork = False  # 割り角
            if forkmove and abs(moved_piece) == BISHOP:
                bishopfork = True
            goldonhead = False  # 頭金
            if ((moved_piece == -GOLD or moved_piece == -TOKIN) and self.dropped_pos+5 == oppking_pos) or \
                    ((moved_piece == GOLD or moved_piece == TOKIN) and self.dropped_pos-5 == oppking_pos):
                goldonhead = True
            goldonbottom = False  # 尻金
            if ((moved_piece == -GOLD or moved_piece == -TOKIN) and self.dropped_pos-5 == oppking_pos) or \
                    ((moved_piece == GOLD or moved_piece == TOKIN) and self.dropped_pos+5 == oppking_pos):
                goldonbottom = True
            silveronknight = False  # 桂頭の銀
            if (moved_piece == -SILVER and self.dropped_pos<20 and self.board[self.dropped_pos+5] == KNIGHT) or \
                    (moved_piece == SILVER and self.dropped_pos>4 and self.board[self.dropped_pos-5] == -KNIGHT):
                silveronknight = True
            silveronbelly = False  # 腹銀
            if abs(moved_piece) == SILVER:
                for w in set(EIGHT_DIR.get(self.dropped_pos, DEF_8_DIR)) & {1, -1}:
                    if self.dropped_pos+w == oppking_pos:
                        silveronbelly = True
            lanceonback = False  # 下段の香車
            if self.selected_pos >= 100 and ((moved_piece == -LANCE and self.dropped_pos in BACKRANK1) or \
                    (moved_piece == LANCE and self.dropped_pos in BACKRANK2)):
                lanceonback = True
            kingenter = False  # 入玉
            if (moved_piece == -KING and self.dropped_pos in BACKRANK2 and not self.selected_pos in BACKRANK2) or \
                    (moved_piece == KING and self.dropped_pos in BACKRANK1 and not self.selected_pos in BACKRANK1):
                kingenter = True
            pawnnogo = False  # 行き所のない歩
            if (moved_piece == -PAWN and 20 <= self.dropped_pos) or \
                    (moved_piece == PAWN and self.dropped_pos < 5):
                pawnnogo = True
            lancenogo = False  # 行き所のない香
            if (moved_piece == -LANCE and 20 <= self.dropped_pos) or \
                    (moved_piece == LANCE and self.dropped_pos < 5):
                lancenogo = True
            knightnogo = False  # 行き所のない桂
            if (moved_piece == -KNIGHT and 15 <= self.dropped_pos) or \
                    (moved_piece == KNIGHT and self.dropped_pos < 10):
                knightnogo = True
            edgeattack = False  # 端攻め
            if abs(moved_piece) == LANCE or abs(moved_piece) == ROOK:
                if (oppking_pos in EDGE1 and self.dropped_pos in EDGE1) or \
                        (oppking_pos in EDGE2 and self.dropped_pos in EDGE2):
                    edgeattack = True
            
            if self.ischeckmate:
                if doublecheck_num > 1:
                    self.msg.in_message('両王手で詰み'+'！'*min(5, self.checkcont[self.turn]), self.turn)
                elif discovd_chk:
                    self.msg.in_message('開き王手で詰み'+'！'*min(4, self.checkcont[self.turn]), self.turn)
                elif goldonhead:
                    self.msg.in_message('頭金で詰み'+'！'*min(6, self.checkcont[self.turn]), self.turn)
                elif goldonbottom:
                    self.msg.in_message('尻金で詰み'+'！'*min(6, self.checkcont[self.turn]), self.turn)
                else:
                    self.msg.in_message('詰み'+'！'*min(9, self.checkcont[self.turn]), self.turn)
                if self.turn == P2 and self.is_man_com:
                    self.msg.in_message('「あなた」の勝ち', P2)
                else:
                    self.msg.in_message('「'+CHARA_NAME[self.chara[self.turn]]+'」の勝ち', self.turn)
                self.end_wait = 0
            elif doublecheck_num:
                if doublecheck_num > 1:
                    self.msg.in_message('両王手'+'！'*min(8, self.checkcont[self.turn]), self.turn)
                elif discovd_chk:
                    self.msg.in_message('開き（あき）王手'+'！'*min(3, self.checkcont[self.turn]), self.turn)
                elif goldonhead:
                    self.msg.in_message('頭金で王手'+'！'*min(6, self.checkcont[self.turn]), self.turn)
                elif goldonbottom:
                    self.msg.in_message('尻金で王手'+'！'*min(6, self.checkcont[self.turn]), self.turn)
                elif edgeattack:
                    self.msg.in_message('端攻めで王手'+'！'*min(5, self.checkcont[self.turn]), self.turn)
                elif self.checkcont[OPP[self.turn]]:
                    self.msg.in_message('逆王手'+'！'*min(8, self.checkcont[self.turn]), self.turn)
                else:
                    self.msg.in_message('王手'+'！'*min(9, self.checkcont[self.turn]), self.turn)
            elif self.is_com_com or (self.is_man_com and self.turn == P1):
                if self.is_brinkmate:
                    self.msg.in_message(random.choice(['お見事', '参りました', '打つ手なし']), self.turn)
                elif self.prev_move_num == 1:
                    if abs(moved_piece) == KING:
                        self.msg.in_message(random.choice(['逃げの一手', '逃げるしかない']), self.turn)
                    else:
                        self.msg.in_message(random.choice(['他に手がない', 'この手しかない']), self.turn)
                elif silveronknight:
                    self.msg.in_message(random.choice(['桂頭（けいとう）の銀', '桂先（けいさき）の銀']), self.turn)
                elif knightfork:
                    self.msg.in_message(random.choice(['ふんどしの桂', '両取りの桂']), self.turn)
                elif silverfork and self.selected_pos >= 100:
                    self.msg.in_message(random.choice(['割り打ちの銀', '両取りの銀']), self.turn)
                elif silverfork:
                    self.msg.in_message(random.choice(['割り銀', '両取りの銀']), self.turn)
                elif bishopfork and self.selected_pos >= 100:
                    self.msg.in_message(random.choice(['割り打ちの角', '両取りの角']), self.turn)
                elif bishopfork:
                    self.msg.in_message(random.choice(['割り角', '両取りの角']), self.turn)
                elif goldonhead:
                    self.msg.in_message('頭金（あたまきん）', self.turn)
                elif goldonbottom:
                    self.msg.in_message('尻金（しりきん）', self.turn)
                elif silveronbelly:
                    self.msg.in_message('腹銀（はらぎん）', self.turn)
                elif edgeattack:
                    self.msg.in_message('端攻め', self.turn)
                elif self.capture_piece:
                    if abs(self.capture_piece[0]) == LANCE or abs(self.capture_piece[0]) == TOKIN:
                        self.msg.in_message(random.choice(['取る', '香と（きょうと）を取る']), self.turn)
                    elif abs(self.capture_piece[0]) == SILVER or abs(self.capture_piece[0]) == BISHOP:
                        self.msg.in_message(random.choice(['取る', '銀角（ぎんかく）を取る']), self.turn)
                    elif abs(self.capture_piece[0]) == GOLD or abs(self.capture_piece[0]) == KNIGHT:
                        self.msg.in_message(random.choice(['取る', '金桂（きんけい）を取る']), self.turn)
                    elif abs(self.capture_piece[0]) == ROOK or abs(self.capture_piece[0]) == PAWN:
                        self.msg.in_message(random.choice(['取る', '飛歩（ひふ）を取る']), self.turn)
                elif forkmove:
                    self.msg.in_message('両取り', self.turn)
                elif abs(moved_piece) == ROOK and self.selected_pos < 100:
                    self.msg.in_message(random.choice(['歩を突く', '飛車に成る', '飛車成り']), self.turn)
                elif lanceonback:
                    self.msg.in_message(random.choice(['下段の香', '香は下段から']), self.turn)
                elif kingenter:
                    self.msg.in_message(random.choice(['最上段に逃げる', '逃げる']), self.turn)
                elif capture_num and captured:
                    self.msg.in_message(random.choice(['ぶつける', '当てる']), self.turn)
                elif pawnnogo:
                    self.msg.in_message(random.choice(['行き所のない歩', '動けない歩']), self.turn)
                elif lancenogo:
                    self.msg.in_message(random.choice(['行き所のない香', '動けない香']), self.turn)
                elif knightnogo:
                    self.msg.in_message(random.choice(['行き所のない桂', '動けない桂']), self.turn)
                elif self.selected_pos >= 100:
                    if abs(moved_piece) == LANCE:
                        self.msg.in_message(random.choice(['打つ', '香車を打つ', '香を打つ']), self.turn)
                    elif abs(moved_piece) == TOKIN:
                        self.msg.in_message(random.choice(['打つ', 'と金を打つ']), self.turn)
                    elif abs(moved_piece) == SILVER:
                        self.msg.in_message(random.choice(['打つ', '銀を打つ']), self.turn)
                    elif abs(moved_piece) == BISHOP:
                        self.msg.in_message(random.choice(['打つ', '角を打つ']), self.turn)
                    elif abs(moved_piece) == GOLD:
                        self.msg.in_message(random.choice(['打つ', '金を打つ']), self.turn)
                    elif abs(moved_piece) == KNIGHT:
                        self.msg.in_message(random.choice(['打つ', '桂馬を打つ', '桂を打つ']), self.turn)
                    elif abs(moved_piece) == ROOK:
                        self.msg.in_message(random.choice(['打つ', '飛車を打つ']), self.turn)
                    elif abs(moved_piece) == PAWN:
                        self.msg.in_message(random.choice(['打つ', '歩を打つ']), self.turn)
                else:
                    self.msg.in_message('')
            self.is_brinkmate = False
            self.turn = OPP[self.turn]
            if self.ischeckmate:
                self.status = STATUS_END
            else:
                self.status = STATUS_START
        
        elif self.status == STATUS_END:
            if self.is_com_com and not RELEASE_CANDIDATE:
                self.end_wait += 1
                if self.end_wait > END_WAIT:
                    self.reset_start()
                    self.status = STATUS_START
            else:
                self.is_continue = (END_X-1<pyxel.mouse_x<END_X+33 and END_Y-1<pyxel.mouse_y<END_Y+ 9)
                self.is_stop     = (END_X-1<pyxel.mouse_x<END_X+33 and END_Y+9<pyxel.mouse_y<END_Y+19)
                if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):  # LEFT_BUTTON_UP
                    if self.is_continue:
                        self.reset_start()
                        self.status = STATUS_START
                    elif self.is_stop:
                        self.reset_start()
                        self.status = STATUS_TITLE
    
    def draw_turnovertable(self):
        for i in range(8):
            pyxel.blt(TURNOVER_X+i*19-(i%2), TURNOVER_Y, 0, (i+2)*16, 0, 15, 16, 1)
        for i in range(4):
            pyxel.text(TURNOVER_X+15+i*38, TURNOVER_Y+6, '=', 0)
    
    def draw_board(self):
        pyxel.rect(BOARD_X, BOARD_Y, 19*5, 19*5, 4)
        for i in range(6):
            pyxel.line(BOARD_X     , BOARD_Y+i*19, BOARD_X+5*19, BOARD_Y+i*19, 0)
            pyxel.line(BOARD_X+i*19, BOARD_Y,      BOARD_X+i*19, BOARD_Y+5*19, 0)
    
    def draw_piece(self):
        for i in range(5):
            for j in range(5):
                p = self.board[i*5+j]
                pyxel.blt(BOARD_X+2+j*19, BOARD_Y+2+i*19, 0, p*16 if p>=0 else -p*16, 0, \
                        15 if p>=0 else -15, 16 if p>=0 else -16, 1)
    
    def draw_stand(self):
        pyxel.rectb(P1_STAND_X  , P1_STAND_Y  , 34, 68, 0)
        pyxel.rect( P1_STAND_X+1, P1_STAND_Y+1, 32, 66, 4)
        pyxel.rectb(P2_STAND_X  , P2_STAND_Y  , 34, 68, 0)
        pyxel.rect( P2_STAND_X+1, P2_STAND_Y+1, 32, 66, 4)
    
    def draw_inhand(self):
        for i, p in enumerate(self.inhand_p1):
            if 0 <= self.selected_pos-100 <= i:
                i += 1
            pyxel.blt(P1_STAND_X+17-(15*(i%2)), P1_STAND_Y+50-(i//2)*16, 0, -p*16, 0, -15, -16, 1)
        for i, p in enumerate(self.inhand_p2):
            if 0 <= self.selected_pos-200 <= i:
                i += 1
            pyxel.blt(P2_STAND_X+ 2+(15*(i%2)), P2_STAND_Y+ 2+(i//2)*16, 0,  p*16, 0,  15,  16, 1)
    
    def draw_select(self, col=9):
        for i in set([v[3] for v in self.allmove]):
            if i < 100:
                pyxel.rect(BOARD_X+(i%5)*19+1, BOARD_Y+(i//5)*19+1, 18, 18, col)
            elif i < 200:
                i -= 100
                pyxel.rect(P1_STAND_X+17-(i%2)*15, P1_STAND_Y+50-(i//2)*16, 15, 16, col)
            else:
                i -= 200
                pyxel.rect(P2_STAND_X+ 2+(i%2)*15, P2_STAND_Y+ 2+(i//2)*16, 15, 16, col)
    
    def draw_drop(self, col_select=9, col_drop=12):
        if self.select_pos < 100:
            pyxel.rect(BOARD_X+(self.select_pos%5)*19+1, BOARD_Y+(self.select_pos//5)*19+1, 18, 18, col_select)
        elif self.select_pos < 200:
            pyxel.rect(P1_STAND_X+17-((self.select_pos-100)%2)*15, P1_STAND_Y+50-((self.select_pos-100)//2)*16, \
                    15, 16, col_select)
        else:
            pyxel.rect(P2_STAND_X+ 2+((self.select_pos-200)%2)*15, P2_STAND_Y+ 2+((self.select_pos-200)//2)*16, \
                    15, 16, col_select)
        for i in range(len(self.allmove)):
            if self.select_pos == self.allmove[i][3] and self.drop_piece == self.allmove[i][0][self.allmove[i][4]]:
                pyxel.rect(BOARD_X+(self.allmove[i][4]%5)*19+1, BOARD_Y+(self.allmove[i][4]//5)*19+1, \
                        18, 18, col_drop)
    
    def draw_prev(self, turn):
        if 0 <= self.selected_pos < 100:
            pyxel.blt(BOARD_X+2+(self.selected_pos%5)*19, BOARD_Y+2+(self.selected_pos//5)*19, \
                    0, 10*16, 0, 15 if turn == P1 else -15, 16 if turn == P1 else -16, 1)
        elif 100 <= self.selected_pos < 200:
            pyxel.blt(P1_STAND_X+17-((self.selected_pos-100)%2)*15, \
                    P1_STAND_Y+50-((self.selected_pos-100)//2)*16, 0, 10*16, 0, -15, -16, 1)
        elif 200 <= self.selected_pos < 300:
            pyxel.blt(P2_STAND_X+ 2+((self.selected_pos-200)%2)*15, \
                    P2_STAND_Y+ 2+((self.selected_pos-200)//2)*16, 0, 10*16, 0,  15,  16, 1)
        if 0 <= self.dropped_pos < 100:
            pyxel.blt(BOARD_X+2+(self.dropped_pos%5)*19, BOARD_Y+2+(self.dropped_pos//5)*19, \
                    0, 10*16, 0, 15 if turn == P1 else -15, 16 if turn == P1 else -16, 1)
    
    def draw_player(self):
        pyxel.rectb(P1_X  , P1_Y  , 26, 26,  7)
        pyxel.rect (P1_X+1, P1_Y+1, 24, 24,  5)
        if self.turn == P1 and self.status == STATUS_COM_COUNT_BRINKMATE:  # 考え中インジケータ
            rate = (24*(self.lp1+1))//len(self.allmove)
            pyxel.rect (P1_X+1, P1_Y+1+(24-rate), 24, rate,  6)
        pyxel.rectb(P2_X  , P2_Y  , 26, 26,  7)
        pyxel.rect (P2_X+1, P2_Y+1, 24, 24,  5)
        if self.turn == P2 and self.status == STATUS_COM_COUNT_BRINKMATE:  # 考え中インジケータ
            rate = (24*(self.lp1+1))//len(self.allmove)
            pyxel.rect (P2_X+1, P2_Y+1+(24-rate), 24, rate,  6)
        dx_p1 = (self.shake_p1%2)*2-1 if self.shake_p1 else 0
        dx_p2 = (self.shake_p2%2)*2-1 if self.shake_p2 else 0
        pyxel.blt(P1_X+5+dx_p1, P1_Y+5, 0, self.chara[P1]*16, 16, -16, 16, 1)
        pyxel.blt(P2_X+5+dx_p2, P2_Y+5, 0, self.chara[P2]*16, 16,  16, 16, 1)
    
    def draw_title(self):
        msg_man_man = ' ひと  対  ひと'
        msg_man_com = '   ＣＰＵ 対 あなた' if self.is_ccstrict else 'ＣＰＵ 対 あなた 高速'
        msg_com_com = '   ＣＰＵ 対 ＣＰＵ' if self.is_ccstrict else 'ＣＰＵ 対 ＣＰＵ 高速'
        for y in range(3):
            for x in range(3):
                if y != 1 or x != 1:
                    msfont.text(BOARD_X+16+x, BOARD_Y+31+y, msg_man_man, 0)
                    msfont.text(BOARD_X+ 4+x, BOARD_Y+43+y, msg_man_com, 0)
                    msfont.text(BOARD_X+ 4+x, BOARD_Y+55+y, msg_com_com, 0)
        msfont.text(BOARD_X+17, BOARD_Y+32, msg_man_man, 7 if self.is_man_man else 1)
        msfont.text(BOARD_X+ 5, BOARD_Y+44, msg_man_com, 7 if self.is_man_com else 1 if self.is_ccstrict else 2)
        msfont.text(BOARD_X+ 5, BOARD_Y+56, msg_com_com, 7 if self.is_com_com else 1 if self.is_ccstrict else 2)
    
    def draw_end(self):
        msg_continue = 'もう一回'
        msg_stop     = 'やめる'
        for y in range(3):
            for x in range(3):
                if y != 1 or x != 1:
                    msfont.text(END_X-1+x, END_Y   +y, msg_continue, 0)
                    msfont.text(END_X+3+x, END_Y+10+y, msg_stop, 0)
        msfont.text(END_X  , END_Y+ 1, msg_continue, 7 if self.is_continue else 1)
        msfont.text(END_X+4, END_Y+11, msg_stop, 7 if self.is_stop else 1)
    
    def draw_message(self):
        pyxel.rectb(MSG_X  , MSG_Y  , 96, 19, 7)
        pyxel.rect( MSG_X+1, MSG_Y+1, 94, 17, 0)
        if self.msg_scrl:
            msfont.text(MSG_X+2+self.spc2, MSG_Y+10-self.msg_scrl, self.msg2, self.col2)
        else:
            msfont.text(MSG_X+2+self.spc1, MSG_Y+2 , self.msg1, self.col1)
            msfont.text(MSG_X+2+self.spc2, MSG_Y+10, self.msg2, self.col2)
    
    def draw_param_p1(self):
        if type(self.dsp_param[P1]) == list:
            for i, p in enumerate(self.dsp_param[P1][:3]):
                pyxel.text(P1_X_PARAM, P1_Y_PARAM+i*6, f'{p:.6f}', 10 if p==self.dsp_select[P1] else 7)
        elif type(self.dsp_param[P1]) == str:
            pyxel.text(P1_X_PARAM, P1_Y_PARAM, self.dsp_param[P1], 7)
    
    def draw_param_p2(self):
        if type(self.dsp_param[P2]) == list:
            for i, p in enumerate(self.dsp_param[P2][:3]):
                pyxel.text(P2_X_PARAM, P2_Y_PARAM+i*6, f'{p:.6f}', 10 if p==self.dsp_select[P2] else 7)
        elif type(self.dsp_param[P2]) == str:
            pyxel.text(P2_X_PARAM, P2_Y_PARAM, self.dsp_param[P2], 7)
    
    def draw(self):
        pyxel.cls(3)
        self.draw_turnovertable()
        self.draw_board()
        self.draw_player()
        self.draw_stand()
        self.msg.draw_message()
        
        if STATUS_COM_S < self.status < STATUS_COM_E or self.status == STATUS_MAN_SELECT:
            self.draw_prev(self.turn)
            self.draw_select()
        elif self.status == STATUS_MAN_DROP:
            self.draw_drop()
        
        self.draw_piece()
        self.draw_inhand()
        
        if self.status == STATUS_TITLE:
            self.draw_title()
            if not RELEASE_CANDIDATE:
                self.draw_param_p1()
                self.draw_param_p2()
        elif self.status == STATUS_END:
            self.draw_prev(self.turn)
            if not self.is_com_com or RELEASE_CANDIDATE:
                self.draw_end()
        
        if not RELEASE_CANDIDATE:
            if self.is_man_com or self.is_com_com:
                self.draw_param_p1()
            if self.is_com_com:
                self.draw_param_p2()

App()

