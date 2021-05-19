# https://leetcode-cn.com/problems/word-transformer-lcci/
# 单词转换：给定开始词和结束词，以及一个词典，每次只改变一个字母，
# 用词典中给定的词为开始词到结束词找到一个转换序列

import collections
from typing import List


# 思路：
# 1. 搜索问题，DFS或BFS，BFS保证最短路径，DFS可能更快
# 2. 单词单字母不同，使用一个 defaultdict 记录单词每一位是 通用的 的记录，查起临近单词更方便
# 3. BFS 记录 path 时可以使用 dict，这样更新的时候方便

class Solution:
    @staticmethod
    def neighbour(a: str, b: str) -> bool:
        flag = False
        for i in range(len(a)):
            if a[i] != b[i]:
                if flag:
                    return False
                else:
                    flag = True
        return flag

    # code rao-zhi-cheng-shang-2
    def DFS(self, beginWord: str, endWord: str, wordList: List[str]) -> List[str]:
        hashmap = collections.defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                hashmap[word[:i] + '*' + word[i + 1:]].append(word)
        stack = [beginWord]
        w_dict = {beginWord: [beginWord]}
        while stack:
            word = stack.pop()
            if word == endWord:
                return w_dict[word]
            for i in range(len(word)):
                if word[:i] + '*' + word[i + 1:] in hashmap:
                    for tmp in hashmap[word[:i] + '*' + word[i + 1:]]:
                        if tmp not in w_dict:
                            w_dict[tmp] = w_dict[word] + [tmp]
                            stack.append(tmp)
        return []

    def BFS(self, beginWord: str, endWord: str, wordList: List[str]) -> List[str]:
        hashmap = collections.defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                hashmap[word[:i] + '*' + word[i + 1:]].append(word)
        queue = collections.deque([beginWord])
        w_dict = {beginWord: [beginWord]}
        while queue:
            word = queue.popleft()
            if word == endWord:
                return w_dict[word]
            for i in range(len(word)):
                if word[:i] + '*' + word[i + 1:] in hashmap:
                    for tmp in hashmap[word[:i] + '*' + word[i + 1:]]:
                        if tmp not in w_dict:
                            w_dict[tmp] = w_dict[word] + [tmp]
                            queue.append(tmp)
        return []

    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[str]:
        return self.DFS(beginWord, endWord, wordList)


beginWord = "cet"
endWord = "ism"
wordList = ["kid","tag","pup","ail","tun","woo","erg","luz","brr","gay","sip","kay","per","val","mes","ohs","now","boa","cet","pal","bar","die","war","hay","eco","pub","lob","rue","fry","lit","rex","jan","cot","bid","ali","pay","col","gum","ger","row","won","dan","rum","fad","tut","sag","yip","sui","ark","has","zip","fez","own","ump","dis","ads","max","jaw","out","btu","ana","gap","cry","led","abe","box","ore","pig","fie","toy","fat","cal","lie","noh","sew","ono","tam","flu","mgm","ply","awe","pry","tit","tie","yet","too","tax","jim","san","pan","map","ski","ova","wed","non","wac","nut","why","bye","lye","oct","old","fin","feb","chi","sap","owl","log","tod","dot","bow","fob","for","joe","ivy","fan","age","fax","hip","jib","mel","hus","sob","ifs","tab","ara","dab","jag","jar","arm","lot","tom","sax","tex","yum","pei","wen","wry","ire","irk","far","mew","wit","doe","gas","rte","ian","pot","ask","wag","hag","amy","nag","ron","soy","gin","don","tug","fay","vic","boo","nam","ave","buy","sop","but","orb","fen","paw","his","sub","bob","yea","oft","inn","rod","yam","pew","web","hod","hun","gyp","wei","wis","rob","gad","pie","mon","dog","bib","rub","ere","dig","era","cat","fox","bee","mod","day","apr","vie","nev","jam","pam","new","aye","ani","and","ibm","yap","can","pyx","tar","kin","fog","hum","pip","cup","dye","lyx","jog","nun","par","wan","fey","bus","oak","bad","ats","set","qom","vat","eat","pus","rev","axe","ion","six","ila","lao","mom","mas","pro","few","opt","poe","art","ash","oar","cap","lop","may","shy","rid","bat","sum","rim","fee","bmw","sky","maj","hue","thy","ava","rap","den","fla","auk","cox","ibo","hey","saw","vim","sec","ltd","you","its","tat","dew","eva","tog","ram","let","see","zit","maw","nix","ate","gig","rep","owe","ind","hog","eve","sam","zoo","any","dow","cod","bed","vet","ham","sis","hex","via","fir","nod","mao","aug","mum","hoe","bah","hal","keg","hew","zed","tow","gog","ass","dem","who","bet","gos","son","ear","spy","kit","boy","due","sen","oaf","mix","hep","fur","ada","bin","nil","mia","ewe","hit","fix","sad","rib","eye","hop","haw","wax","mid","tad","ken","wad","rye","pap","bog","gut","ito","woe","our","ado","sin","mad","ray","hon","roy","dip","hen","iva","lug","asp","hui","yak","bay","poi","yep","bun","try","lad","elm","nat","wyo","gym","dug","toe","dee","wig","sly","rip","geo","cog","pas","zen","odd","nan","lay","pod","fit","hem","joy","bum","rio","yon","dec","leg","put","sue","dim","pet","yaw","nub","bit","bur","sid","sun","oil","red","doc","moe","caw","eel","dix","cub","end","gem","off","yew","hug","pop","tub","sgt","lid","pun","ton","sol","din","yup","jab","pea","bug","gag","mil","jig","hub","low","did","tin","get","gte","sox","lei","mig","fig","lon","use","ban","flo","nov","jut","bag","mir","sty","lap","two","ins","con","ant","net","tux","ode","stu","mug","cad","nap","gun","fop","tot","sow","sal","sic","ted","wot","del","imp","cob","way","ann","tan","mci","job","wet","ism","err","him","all","pad","hah","hie","aim","ike","jed","ego","mac","baa","min","com","ill","was","cab","ago","ina","big","ilk","gal","tap","duh","ola","ran","lab","top","gob","hot","ora","tia","kip","han","met","hut","she","sac","fed","goo","tee","ell","not","act","gil","rut","ala","ape","rig","cid","god","duo","lin","aid","gel","awl","lag","elf","liz","ref","aha","fib","oho","tho","her","nor","ace","adz","fun","ned","coo","win","tao","coy","van","man","pit","guy","foe","hid","mai","sup","jay","hob","mow","jot","are","pol","arc","lax","aft","alb","len","air","pug","pox","vow","got","meg","zoe","amp","ale","bud","gee","pin","dun","pat","ten","mob"]

s = Solution()
import time
res = []
t1 = time.time()
for i in range(10):
    res = s.findLadders(beginWord, endWord, wordList)
t2 = time.time()
print(t2-t1, res)
