# https://leetcode-cn.com/problems/word-transformer-lcci/
# 单词转换：给定开始词和结束词，以及一个词典，每次只改变一个字母，
# 用词典中给定的词为开始词到结束词找到一个转换序列

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

    @staticmethod
    def HamingDist(a: str, b: str) -> int:
        count = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                count += 1
        return count

    @staticmethod
    def recur(current, dist, candids, endWord):
        if Solution.neighbour(current, endWord):
            return True, [current, endWord]
        for w in list(candids.keys()):
            d = candids[w]
            if Solution.neighbour(current, w):
                if d > dist:
                    continue
                del candids[w]
                status, ladder = Solution.recur(w, d, candids, endWord)
                if status:
                    return True, [current] + ladder
                candids[w] = d
        return False, []

    # def findLadders(self, beginWord, endWord, wordList):
    #     if endWord not in wordList:
    #         return []
    #     if Solution.neighbour(beginWord, endWord):
    #         return [beginWord, endWord]
    #     wordList.remove(endWord)
    #     candids = {w: Solution.HamingDist(endWord, w) for w in wordList}
    #     status, ladder = \
    #         Solution.recur(beginWord, Solution.HamingDist(beginWord, endWord), candids, endWord)
    #     return ladder

    # def dfs(self, curWord: str, endWord: str, wordList) -> bool:
    #     if curWord == endWord:
    #         return True
    #     n = len(wordList)
    #     for i in range(n):
    #         if self.visited[i] or not Solution.neighbour(curWord, wordList[i]):
    #             continue
    #         self.visited[i] = True
    #         self.path.append(wordList[i])
    #         if self.dfs(wordList[i], endWord, wordList):
    #             return True
    #         self.path.pop(-1)
    #     return False

    # def findLadders(self, beginWord: str, endWord: str, wordList):
    #     n = len(wordList)
    #     self.visited = [False for _ in range(n)]
    #     self.path = [beginWord]
    #     if self.dfs(beginWord, endWord, wordList):
    #         return self.path[:]
    #     else:
    #         return []

    @staticmethod
    def bfs(current, candids):
        for w, v in candids.items():
            layer = []
            if v and Solution.neighbour(current, w):
                layer.append(w)




# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log","cog"]

beginWord = "cet"
endWord = "ism"
wordList = ["kid","tag","pup","ail","tun","woo","erg","luz","brr","gay","sip","kay","per","val","mes","ohs","now","boa","cet","pal","bar","die","war","hay","eco","pub","lob","rue","fry","lit","rex","jan","cot","bid","ali","pay","col","gum","ger","row","won","dan","rum","fad","tut","sag","yip","sui","ark","has","zip","fez","own","ump","dis","ads","max","jaw","out","btu","ana","gap","cry","led","abe","box","ore","pig","fie","toy","fat","cal","lie","noh","sew","ono","tam","flu","mgm","ply","awe","pry","tit","tie","yet","too","tax","jim","san","pan","map","ski","ova","wed","non","wac","nut","why","bye","lye","oct","old","fin","feb","chi","sap","owl","log","tod","dot","bow","fob","for","joe","ivy","fan","age","fax","hip","jib","mel","hus","sob","ifs","tab","ara","dab","jag","jar","arm","lot","tom","sax","tex","yum","pei","wen","wry","ire","irk","far","mew","wit","doe","gas","rte","ian","pot","ask","wag","hag","amy","nag","ron","soy","gin","don","tug","fay","vic","boo","nam","ave","buy","sop","but","orb","fen","paw","his","sub","bob","yea","oft","inn","rod","yam","pew","web","hod","hun","gyp","wei","wis","rob","gad","pie","mon","dog","bib","rub","ere","dig","era","cat","fox","bee","mod","day","apr","vie","nev","jam","pam","new","aye","ani","and","ibm","yap","can","pyx","tar","kin","fog","hum","pip","cup","dye","lyx","jog","nun","par","wan","fey","bus","oak","bad","ats","set","qom","vat","eat","pus","rev","axe","ion","six","ila","lao","mom","mas","pro","few","opt","poe","art","ash","oar","cap","lop","may","shy","rid","bat","sum","rim","fee","bmw","sky","maj","hue","thy","ava","rap","den","fla","auk","cox","ibo","hey","saw","vim","sec","ltd","you","its","tat","dew","eva","tog","ram","let","see","zit","maw","nix","ate","gig","rep","owe","ind","hog","eve","sam","zoo","any","dow","cod","bed","vet","ham","sis","hex","via","fir","nod","mao","aug","mum","hoe","bah","hal","keg","hew","zed","tow","gog","ass","dem","who","bet","gos","son","ear","spy","kit","boy","due","sen","oaf","mix","hep","fur","ada","bin","nil","mia","ewe","hit","fix","sad","rib","eye","hop","haw","wax","mid","tad","ken","wad","rye","pap","bog","gut","ito","woe","our","ado","sin","mad","ray","hon","roy","dip","hen","iva","lug","asp","hui","yak","bay","poi","yep","bun","try","lad","elm","nat","wyo","gym","dug","toe","dee","wig","sly","rip","geo","cog","pas","zen","odd","nan","lay","pod","fit","hem","joy","bum","rio","yon","dec","leg","put","sue","dim","pet","yaw","nub","bit","bur","sid","sun","oil","red","doc","moe","caw","eel","dix","cub","end","gem","off","yew","hug","pop","tub","sgt","lid","pun","ton","sol","din","yup","jab","pea","bug","gag","mil","jig","hub","low","did","tin","get","gte","sox","lei","mig","fig","lon","use","ban","flo","nov","jut","bag","mir","sty","lap","two","ins","con","ant","net","tux","ode","stu","mug","cad","nap","gun","fop","tot","sow","sal","sic","ted","wot","del","imp","cob","way","ann","tan","mci","job","wet","ism","err","him","all","pad","hah","hie","aim","ike","jed","ego","mac","baa","min","com","ill","was","cab","ago","ina","big","ilk","gal","tap","duh","ola","ran","lab","top","gob","hot","ora","tia","kip","han","met","hut","she","sac","fed","goo","tee","ell","not","act","gil","rut","ala","ape","rig","cid","god","duo","lin","aid","gel","awl","lag","elf","liz","ref","aha","fib","oho","tho","her","nor","ace","adz","fun","ned","coo","win","tao","coy","van","man","pit","guy","foe","hid","mai","sup","jay","hob","mow","jot","are","pol","arc","lax","aft","alb","len","air","pug","pox","vow","got","meg","zoe","amp","ale","bud","gee","pin","dun","pat","ten","mob"]

s = Solution()
print(s.findLadders(beginWord, endWord, wordList))
