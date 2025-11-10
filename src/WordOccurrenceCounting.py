'''

AA - Trabalho 2
Estudo de Contagens Aproximadas de Ocorrências de Palavras de Livros (traduzidos em várias línguas)
Autor: Filipe Pires
NMEC: 85122
Data: 12/2019

'''

################################################################################ Required Libraries ##################################################################

from collections import Counter
import re

import math
import random

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil

import nltk
from nltk.corpus import stopwords 

from sys import getsizeof

################################################################################ Data ################################################################################

# books to be studied and title/language abbreviations
Books = [] 
titleAbbreviations = {}
languageAbbreviations = {}

# top results to be presented and errors/deviations associated to the approximate counters
results = {}
deviations = {} # maximal, mean absolute and standard deviations + error percentage and average error percentage

# total number of words by book and language and total number of words actually considered by book and language
numWords = {} 
numTokens = {} 

# words to be ignored (by language)
StopWords = {}

K = 10 # number of words with highest counts to return (by all counters)
N = 10 # number of times approximate counters are executed

FixedProb = 0.5 # counting probability for approxCountingFixedProb()
LogBase = math.sqrt(2) # counting logarithmic base for approxCountingLogarithmic()

Out = "../results/" # folder where the results will be stored (in .csv and formatted .txt)


################################################################################ Counting ##########################################################################

def exactCounting(books,k,study=0):
    global StopWords
    global results
    global numWords, numTokens

    exactCtr = Counter()
    print(nltk.corpus.stopwords.words("english"))
    for b in books:
        #print("\n" + b[0])
        for lang in b[1]:
            #print(lang[0])
            f = open(lang[1], "r", encoding="utf-8")
            for line in f:
                words = re.findall(r'\w+',line)
                for word in words:
                    if lang[0] in numWords[b[0]].keys():
                        numWords[b[0]][lang[0]] += 1
                    else:
                        numWords[b[0]][lang[0]] = 1
                    if len(word) > 2 and word.lower() not in StopWords[lang[0]]:
                        exactCtr.update([word])
            numTokens[b[0]][lang[0]] = len(list(exactCtr.elements()))
            if study != 0:
                results["Exact Counter"][b[0]][lang[0]] = exactCtr
            else:
                results["Exact Counter"][b[0]][lang[0]] = exactCtr.most_common(k)
            f.close()
            exactCtr = Counter()
    return

        
def approxCountingFixedProb(books,k,prob,study=0):
    global StopWords
    global results
    
    approxCtrFix = Counter()
    counterName = "Approximate Counter w/ Fixed Prob"
    
    for b in books:
        #print("\n" + b[0])
        for lang in b[1]:
            #print(lang[0])
            f = open(lang[1], "r", encoding="utf-8")
            for line in f:
                words = re.findall(r'\w+',line)
                for word in words:
                    if len(word) > 2 and word.lower() not in StopWords[lang[0]]:
                        if approxCtrFix[word]:
                            if random.randint(1,100)<=100*prob:
                                approxCtrFix.update([word])
                        else:
                            approxCtrFix.update([word])
            
            if study != 0:
                if lang[0] not in results[counterName][b[0]].keys():
                    results[counterName][b[0]][lang[0]] = {}
                if study in results[counterName][b[0]][lang[0]].keys():
                    approxCtrFix.update(results[counterName][b[0]][lang[0]][study])
                results[counterName][b[0]][lang[0]][study] = approxCtrFix
            else:
                results[counterName][b[0]][lang[0]] = approxCtrFix.most_common(k)

            f.close()
            approxCtrFix = Counter()
    return


def approxCountingLogarithmic(books,k,base,study=0):
    global StopWords
    global results
    
    approxCtrLog = Counter()
    counterName = "Approximate Counter w/ Logarithmic Prob"
    
    for b in books:
        #print("\n" + b[0])
        for lang in b[1]:
            #print(lang[0])
            f = open(lang[1], "r",encoding="utf-8")
            for line in f:
                words = re.findall(r'\w+',line)
                for word in words:
                    if len(word) > 2 and word.lower() not in StopWords[lang[0]]:
                        if approxCtrLog[word]:
                            prob = 1 / (base**approxCtrLog[word])
                            if random.randint(1,100)<=100*prob:
                                approxCtrLog.update([word])
                        else:
                            approxCtrLog.update([word])
                            
            if study != 0:
                if lang[0] not in results[counterName][b[0]].keys():
                    results[counterName][b[0]][lang[0]] = {}
                if study in results[counterName][b[0]][lang[0]].keys():
                    approxCtrLog.update(results[counterName][b[0]][lang[0]][study])
                results[counterName][b[0]][lang[0]][study] = approxCtrLog
            else:
                results[counterName][b[0]][lang[0]] = approxCtrLog.most_common(k)

            f.close()
            approxCtrLog = Counter()
    return


################################################################################ Auxiliary Functions #################################################################

def prepareGlobalVariables():
    global Out
    global Books, StopWords
    global titleAbbreviations, languageAbbreviations
    global results, deviations
    global numWords, numTokens

    if os.path.exists(Out):
        shutil.rmtree(Out, ignore_errors=True)
    os.makedirs(Out)
    os.makedirs(Out + "charts")
    os.makedirs(Out + "plots")

    b1 = [("EN","../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"),
            ("FI","../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_Finnish.txt"),
            ("GE","../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_German.txt"),
            ("DU","../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_Dutch.txt"),
            ("FR","../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_French.txt")]
    b2 = [("EN","../datasets/KingSolomonsMines_HRiderHaggard/KingSolomonsMines_HRiderHaggard_English.txt"),
            ("FI","../datasets/KingSolomonsMines_HRiderHaggard/KingSolomonsMines_HRiderHaggard_Finnish.txt"),
            ("PT","../datasets/KingSolomonsMines_HRiderHaggard/KingSolomonsMines_HRiderHaggard_Portuguese.txt")]
    b3 = [("EN","../datasets/OliverTwist_CharlesDickens/OliverTwist_CharlesDickens_English.txt"),
            ("GE","../datasets/OliverTwist_CharlesDickens/OliverTwist_CharlesDickens_German.txt"),
            ("FR","../datasets/OliverTwist_CharlesDickens/OliverTwist_CharlesDickens_French.txt")]
    b4 = [("EN","../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt"),
            ("FI","../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_Finnish.txt"),
            ("GE","../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_German.txt"),
            ("CA","../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_Catalan.txt")]
    Books.append(("ACC",b1))
    Books.append(("KSM",b2))
    Books.append(("OT",b3))
    Books.append(("TATS",b4))

    results["Exact Counter"] = {}
    results["Approximate Counter w/ Fixed Prob"] = {}
    results["Approximate Counter w/ Logarithmic Prob"] = {}
    for ctrKey in results.keys():
        results[ctrKey]["ACC"] = {}
        results[ctrKey]["KSM"] = {}
        results[ctrKey]["OT"] = {}
        results[ctrKey]["TATS"] = {}

    deviations["Approximate Counter w/ Fixed Prob"] = {}
    deviations["Approximate Counter w/ Logarithmic Prob"] = {}
    for ctrKey in deviations.keys():
        deviations[ctrKey]["ACC"] = {}
        deviations[ctrKey]["KSM"] = {}
        deviations[ctrKey]["OT"] = {}
        deviations[ctrKey]["TATS"] = {}

    numWords["ACC"] = {}
    numWords["KSM"] = {}
    numWords["OT"] = {}
    numWords["TATS"] = {}

    numTokens["ACC"] = {}
    numTokens["KSM"] = {}
    numTokens["OT"] = {}
    numTokens["TATS"] = {}

    titleAbbreviations["ACC"] = "A Christmas Carol, by Charles Dickens"
    titleAbbreviations["KSM"] = "King Solomon's Mines, by H. Rider Haggard"
    titleAbbreviations["OT"] = "Oliver Twist, by Charles Dickens"
    titleAbbreviations["TATS"] = "The Adventures Of Tom Sawyer, by Mark Twain"

    languageAbbreviations["EN"] = "English"
    languageAbbreviations["FI"] = "Finnish"
    languageAbbreviations["GE"] = "German"
    languageAbbreviations["DU"] = "Dutch"
    languageAbbreviations["FR"] = "French"
    languageAbbreviations["PT"] = "Portuguese"
    languageAbbreviations["CA"] = "Catalan"

    # StopWords = {"EN":["ll","tis","twas","ve","a","s","able","ableabout","about","above","abroad","abst","accordance","according","accordingly","across","act","actually","ad","added","adj","adopted","ae","af","affected","affecting","affects","after","afterwards","ag","again","against","ago","ah","ahead","ai","ain","t","aint","al","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","amoungst","amount","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","ao","apart","apparently","appear","appreciate","appropriate","approximately","aq","ar","are","area","areas","aren","t","arent","arise","around","arpa","as","aside","ask","asked","asking","asks","associated","at","au","auth","available","aw","away","awfully","az","b","ba","back","backed","backing","backs","backward","backwards","bb","bd","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","bf","bg","bh","bi","big","bill","billion","biol","bj","bm","bn","bo","both","bottom","br","brief","briefly","bs","bt","but","buy","bv","bw","by","bz","c","mon","ca","call","came","can","cannot","cant","caption","case","cases","cause","causes","cc","cd","certain","certainly","cf","cg","ch","changes","ci","ck","cl","clear","clearly","click","cm","cmon","cn","co","co.","com","come","comes","computer","con","concerning","consequently","consider","considering","contain","containing","contains","copy","corresponding","could","ve","couldn","couldnt","course","cr","cry","cs","cu","currently","cv","cx","cy","cz","d","dare","daren","darent","date","de","dear","definitely","describe","described","despite","detail","did","didn","didnt","differ","different","differently","directly","dj","dk","dm","do","does","doesn","doesnt","doing","don","done","dont","doubtful","down","downed","downing","downs","downwards","due","during","dz","e","each","early","ec","ed","edu","ee","effect","eg","eh","eight","eighty","either","eleven","else","elsewhere","empty","end","ended","ending","ends","enough","entirely","er","es","especially","et","et-al","etc","even","evenly","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","fairly","far","farther","felt","few","fewer","ff","fi","fifteen","fifth","fifty","fify","fill","find","finds","fire","first","five","fix","fj","fk","fm","fo","followed","following","follows","for","forever","former","formerly","forth","forty","forward","found","four","fr","free","from","front","full","fully","further","furthered","furthering","furthermore","furthers","fx","g","ga","gave","gb","gd","ge","general","generally","get","gets","getting","gf","gg","gh","gi","give","given","gives","giving","gl","gm","gmt","gn","go","goes","going","gone","good","goods","got","gotten","gov","gp","gq","gr","great","greater","greatest","greetings","group","grouped","grouping","groups","gs","gt","gu","gw","gy","h","had","hadn","hadnt","half","happens","hardly","has","hasn","hasnt","have","haven","havent","having","he","d","hed","hell","hello","help","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","herse”","hes","hi","hid","high","higher","highest","him","himself","himse”","his","hither","hk","hm","hn","home","homepage","hopefully","how","howbeit","however","hr","ht","htm","html","http","hu","hundred","i","i.e.","id","ie","if","ignored","ii","il","ill","im","immediate","immediately","importance","important","in","inasmuch","inc","inc.","indeed","index","indicate","indicated","indicates","information","inner","inside","insofar","instead","int","interest","interested","interesting","interests","into","invention","inward","io","iq","ir","is","isn","isnt","it","itd","itll","its","itself","itse”","ive","j","je","jm","jo","join","jp","just","k","ke","keep","keeps","kept","keys","kg","kh","ki","kind","km","kn","knew","know","known","knows","kp","kr","kw","ky","kz","l","la","large","largely","last","lately","later","latest","latter","latterly","lb","lc","least","length","less","lest","let","lets","li","like","liked","likely","likewise","line","little","lk","ll","long","longer","longest","look","looking","looks","low","lower","lr","ls","lt","ltd","lu","lv","ly","m","ma","made","mainly","make","makes","making","man","many","may","maybe","maynt","mc","md","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","mh","microsoft","might","mightn","mightnt","mil","mill","million","mine","minus","miss","mk","ml","mm","mn","mo","more","moreover","most","mostly","move","mp","mq","mr","mrs","ms","msie","mt","mu","much","mug","must","mustn","mustnt","mv","mw","mx","my","myself","myse”","mz","n","na","name","namely","nay","nc","nd","ne","near","nearly","necessarily","necessary","need","needed","needing","needn","neednt","needs","neither","net","netscape","never","neverf","neverless","nevertheless","new","newer","newest","next","nf","ng","ni","nine","ninety","nl","no","no-one","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","notwithstanding","novel","now","nowhere","np","nr","nu","null","number","numbers","nz","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","om","omitted","on","once","one","ones","only","onto","open","opened","opening","opens","opposite","or","ord","order","ordered","ordering","orders","org","other","others","otherwise","ought","oughtn","oughtnt","our","ours","ourselves","out","outside","over","overall","owing","own","p","pa","page","pages","part","parted","particular","particularly","parting","parts","past","pe","per","perhaps","pf","pg","ph","pk","pl","place","placed","places","please","plus","pm","pmid","pn","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","pr","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provided","provides","pt","put","puts","pw","py","q","qa","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","ref","refs","ro","ru","rw","s","sa","sb","sc","sd","se","sec","second","secondly","self","selves","seven","seventy","sg","sh","shall","shan","shant","she","shed","shell","shes","should","shouldn","shouldnt","si","since","six","sixty","sj","sk","sl","sm","sn","so","some","somebody","someday","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","sr","st","still","su","sub","such","sup","sure","sv","sy","sz","t","tc","td","ten","tf","tg","th","than","thanx","that","thatll","thats","thatve","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","thered","therefore","therein","therell","thereof","therere","theres","thereto","thereupon","thereve","these","they","theyd","theyll","theyre","theyve","third","thirty","this","thorough","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","till","tip","tis","tj","tk","tm","tn","to","too","tp","tr","trillion","ts","tt","tw","twas","twelve","twenty","twice","two","tz","u","ua","ug","uk","um","un","upon","ups","uucp","uy","uz","v","va","vc","ve","vg","vi","via","viz","vn","vol","vols","vs","vu","w","want","wanted","wanting","wants","was","wasn","wasnt","way","ways","we","web","webpage","website","wed","welcome","well","wells","went","were","weren","weren","werent","weve","wf","what","whatever","whatll","whats","whatve","when","whence","whenever","where","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","whichever","while","whilst","whim","whither","who","whod","whoever","whole","wholl","whom","whomever","whos","whose","why","widely","width","will","willing","wish","with","within","without","won","wonder","wont","words","work","worked","working","works","world","would","wouldn","wouldnt","ws","www","x","y","ye","year","years","yes","yet","you","youd","youll","young","younger","youngest","your","youre","yours","yourself","yourselves","youve","yt","yu","z","za","zero","zm","zr"],
    #     "FI":["aiemmin","aika","aikaa","aikaan","aikaisemmin","aikaisin","aikajen","aikana","aikoina","aikoo","aikovat","aina","ainakaan","ainakin","ainoa","ainoat","aiomme","aion","aiotte","aist","aivan","ajan","alas","alemmas","alkuisin","alkuun","alla","alle","aloitamme","aloitan","aloitat","aloitatte","aloitattivat","aloitettava","aloitettevaksi","aloitettu","aloitimme","aloitin","aloitit","aloititte","aloittaa","aloittamatta","aloitti","aloittivat","alta","aluksi","alussa","alusta","annettavaksi","annetteva","annettu","ansiosta","antaa","antamatta","antoi","aoua","apu","asia","asiaa","asian","asiasta","asiat","asioiden","asioihin","asioita","asti","avuksi","avulla","avun","avutta","edelle","edelleen","edellä","edeltä","edemmäs","edes","edessä","edestä","ehkä","ei","eikä","eilen","eivät","eli","ellei","elleivät","ellemme","ellen","ellet","ellette","emme","en","enemmän","eniten","ennen","ensi","ensimmäinen","ensimmäiseksi","ensimmäisen","ensimmäisenä","ensimmäiset","ensimmäisiksi","ensimmäisinä","ensimmäisiä","ensimmäistä","ensin","entinen","entisen","entisiä","entisten","entistä","enää","eri","erittäin","erityisesti","eräiden","eräs","eräät","esi","esiin","esillä","esimerkiksi","et","eteen","etenkin","etessa","ette","ettei","että","haikki","halua","haluaa","haluamatta","haluamme","haluan","haluat","haluatte","haluavat","halunnut","halusi","halusimme","halusin","halusit","halusitte","halusivat","halutessa","haluton","he","hei","heidän","heidät","heihin","heille","heillä","heiltä","heissä","heistä","heitä","helposti","heti","hetkellä","hieman","hitaasti","hoikein","huolimatta","huomenna","hyvien","hyviin","hyviksi","hyville","hyviltä","hyvin","hyvinä","hyvissä","hyvistä","hyviä","hyvä","hyvät","hyvää","hän","häneen","hänelle","hänellä","häneltä","hänen","hänessä","hänestä","hänet","häntä","ihan","ilman","ilmeisesti","itse","itsensä","itseään","ja","jo","johon","joiden","joihin","joiksi","joilla","joille","joilta","joina","joissa","joista","joita","joka","jokainen","jokin","joko","joksi","joku","jolla","jolle","jolloin","jolta","jompikumpi","jona","jonka","jonkin","jonne","joo","jopa","jos","joskus","jossa","josta","jota","jotain","joten","jotenkin","jotenkuten","jotka","jotta","jouduimme","jouduin","jouduit","jouduitte","joudumme","joudun","joudutte","joukkoon","joukossa","joukosta","joutua","joutui","joutuivat","joutumaan","joutuu","joutuvat","juuri","jälkeen","jälleen","jää","kahdeksan","kahdeksannen","kahdella","kahdelle","kahdelta","kahden","kahdessa","kahdesta","kahta","kahteen","kai","kaiken","kaikille","kaikilta","kaikkea","kaikki","kaikkia","kaikkiaan","kaikkialla","kaikkialle","kaikkialta","kaikkien","kaikkin","kaksi","kannalta","kannattaa","kanssa","kanssaan","kanssamme","kanssani","kanssanne","kanssasi","kauan","kauemmas","kaukana","kautta","kehen","keiden","keihin","keiksi","keille","keillä","keiltä","keinä","keissä","keistä","keitten","keittä","keitä","keneen","keneksi","kenelle","kenellä","keneltä","kenen","kenenä","kenessä","kenestä","kenet","kenettä","kennessästä","kenties","kerran","kerta","kertaa","keskellä","kesken","keskimäärin","ketkä","ketä","kiitos","kohti","koko","kokonaan","kolmas","kolme","kolmen","kolmesti","koska","koskaan","kovin","kuin","kuinka","kuinkan","kuitenkaan","kuitenkin","kuka","kukaan","kukin","kukka","kumpainen","kumpainenkaan","kumpi","kumpikaan","kumpikin","kun","kuten","kuuden","kuusi","kuutta","kylliksi","kyllä","kymmenen","kyse","liian","liki","lisäksi","lisää","lla","luo","luona","lähekkäin","lähelle","lähellä","läheltä","lähemmäs","lähes","lähinnä","lähtien","läpi","mahdollisimman","mahdollista","me","meidän","meidät","meihin","meille","meillä","meiltä","meissä","meistä","meitä","melkein","melko","menee","meneet","menemme","menen","menet","menette","menevät","meni","menimme","menin","menit","menivät","mennessä","mennyt","menossa","mihin","mikin","miksi","mikä","mikäli","mikään","mille","milloin","milloinkan","millä","miltä","minkä","minne","minua","minulla","minulle","minulta","minun","minussa","minusta","minut","minuun","minä","missä","mistä","miten","mitkä","mitä","mitään","moi","molemmat","mones","monesti","monet","moni","moniaalla","moniaalle","moniaalta","monta","muassa","muiden","muita","muka","mukaan","mukaansa","mukana","mutta","muu","muualla","muualle","muualta","muuanne","muulloin","muun","muut","muuta","muutama","muutaman","muuten","myöhemmin","myös","myöskin","myöskään","myötä","ne","neljä","neljän","neljää","niiden","niihin","niiksi","niille","niillä","niiltä","niin","niinä","niissä","niistä","niitä","noiden","noihin","noiksi","noilla","noille","noilta","noin","noina","noissa","noista","noita","nopeammin","nopeasti","nopeiten","nro","nuo","nyt","näiden","näihin","näiksi","näille","näillä","näiltä","näin","näinä","näissä","näissähin","näissälle","näissältä","näissästä","näistä","näitä","nämä","ohi","oikea","oikealla","oikein","ole","olemme","olen","olet","olette","oleva","olevan","olevat","oli","olimme","olin","olisi","olisimme","olisin","olisit","olisitte","olisivat","olit","olitte","olivat","olla","olleet","olli","ollut","oma","omaa","omaan","omaksi","omalle","omalta","oman","omassa","omat","omia","omien","omiin","omiksi","omille","omilta","omissa","omista","on","onkin","onko","ovat","paikoittain","paitsi","pakosti","paljon","paremmin","parempi","parhaillaan","parhaiten","perusteella","peräti","pian","pieneen","pieneksi","pienelle","pienellä","pieneltä","pienempi","pienestä","pieni","pienin","poikki","puolesta","puolestaan","päälle","runsaasti","saakka","sadam","sama","samaa","samaan","samalla","samallalta","samallassa","samallasta","saman","samat","samoin","sata","sataa","satojen","se","seitsemän","sekä","sen","seuraavat","siellä","sieltä","siihen","siinä","siis","siitä","sijaan","siksi","sille","silloin","sillä","silti","siltä","sinne","sinua","sinulla","sinulle","sinulta","sinun","sinussa","sinusta","sinut","sinuun","sinä","sisäkkäin","sisällä","siten","sitten","sitä","ssa","sta","suoraan","suuntaan","suuren","suuret","suuri","suuria","suurin","suurten","taa","taas","taemmas","tahansa","tai","takaa","takaisin","takana","takia","tallä","tapauksessa","tarpeeksi","tavalla","tavoitteena","te","teidän","teidät","teihin","teille","teillä","teiltä","teissä","teistä","teitä","tietysti","todella","toinen","toisaalla","toisaalle","toisaalta","toiseen","toiseksi","toisella","toiselle","toiselta","toisemme","toisen","toisensa","toisessa","toisesta","toista","toistaiseksi","toki","tosin","tuhannen","tuhat","tule","tulee","tulemme","tulen","tulet","tulette","tulevat","tulimme","tulin","tulisi","tulisimme","tulisin","tulisit","tulisitte","tulisivat","tulit","tulitte","tulivat","tulla","tulleet","tullut","tuntuu","tuo","tuohon","tuoksi","tuolla","tuolle","tuolloin","tuolta","tuon","tuona","tuonne","tuossa","tuosta","tuota","tuotä","tuskin","tykö","tähän","täksi","tälle","tällä","tällöin","tältä","tämä","tämän","tänne","tänä","tänään","tässä","tästä","täten","tätä","täysin","täytyvät","täytyy","täällä","täältä","ulkopuolella","usea","useasti","useimmiten","usein","useita","uudeksi","uudelleen","uuden","uudet","uusi","uusia","uusien","uusinta","uuteen","uutta","vaan","vahemmän","vai","vaiheessa","vaikea","vaikean","vaikeat","vaikeilla","vaikeille","vaikeilta","vaikeissa","vaikeista","vaikka","vain","varmasti","varsin","varsinkin","varten","vasen","vasenmalla","vasta","vastaan","vastakkain","vastan","verran","vielä","vierekkäin","vieressä","vieri","viiden","viime","viimeinen","viimeisen","viimeksi","viisi","voi","voidaan","voimme","voin","voisi","voit","voitte","voivat","vuoden","vuoksi","vuosi","vuosien","vuosina","vuotta","vähemmän","vähintään","vähiten","vähän","välillä","yhdeksän","yhden","yhdessä","yhteen","yhteensä","yhteydessä","yhteyteen","yhtä","yhtäälle","yhtäällä","yhtäältä","yhtään","yhä","yksi","yksin","yksittäin","yleensä","ylemmäs","yli","ylös","ympäri","älköön","älä"],
    #     "GE":["a","ab","aber","ach","acht","achte","achten","achter","achtes","ag","alle","allein","allem","allen","aller","allerdings","alles","allgemeinen","als","also","am","an","ander","andere","anderem","anderen","anderer","anderes","anderm","andern","anderr","anders","au","auch","auf","aus","ausser","ausserdem","außer","außerdem","b","bald","bei","beide","beiden","beim","beispiel","bekannt","bereits","besonders","besser","besten","bin","bis","bisher","bist","c","d","d.h","da","dabei","dadurch","dafür","dagegen","daher","dahin","dahinter","damals","damit","danach","daneben","dank","dann","daran","darauf","daraus","darf","darfst","darin","darum","darunter","darüber","das","dasein","daselbst","dass","dasselbe","davon","davor","dazu","dazwischen","daß","dein","deine","deinem","deinen","deiner","deines","dem","dementsprechend","demgegenüber","demgemäss","demgemäß","demselben","demzufolge","den","denen","denn","denselben","der","deren","derer","derjenige","derjenigen","dermassen","dermaßen","derselbe","derselben","des","deshalb","desselben","dessen","deswegen","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","dir","doch","dort","drei","drin","dritte","dritten","dritter","drittes","du","durch","durchaus","durfte","durften","dürfen","dürft","e","eben","ebenso","ehrlich","ei","ei,","eigen","eigene","eigenen","eigener","eigenes","ein","einander","eine","einem","einen","einer","eines","einig","einige","einigem","einigen","einiger","einiges","einmal","eins","elf","en","ende","endlich","entweder","er","ernst","erst","erste","ersten","erster","erstes","es","etwa","etwas","euch","euer","eure","eurem","euren","eurer","eures","f","folgende","früher","fünf","fünfte","fünften","fünfter","fünftes","für","g","gab","ganz","ganze","ganzen","ganzer","ganzes","gar","gedurft","gegen","gegenüber","gehabt","gehen","geht","gekannt","gekonnt","gemacht","gemocht","gemusst","genug","gerade","gern","gesagt","geschweige","gewesen","gewollt","geworden","gibt","ging","gleich","gott","gross","grosse","grossen","grosser","grosses","groß","große","großen","großer","großes","gut","gute","guter","gutes","h","hab","habe","haben","habt","hast","hat","hatte","hatten","hattest","hattet","heisst","her","heute","hier","hin","hinter","hoch","hätte","hätten","i","ich","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","im","immer","in","indem","infolgedessen","ins","irgend","ist","j","ja","jahr","jahre","jahren","je","jede","jedem","jeden","jeder","jedermann","jedermanns","jedes","jedoch","jemand","jemandem","jemanden","jene","jenem","jenen","jener","jenes","jetzt","k","kam","kann","kannst","kaum","kein","keine","keinem","keinen","keiner","keines","kleine","kleinen","kleiner","kleines","kommen","kommt","konnte","konnten","kurz","können","könnt","könnte","l","lang","lange","leicht","leide","lieber","los","m","machen","macht","machte","mag","magst","mahn","mal","man","manche","manchem","manchen","mancher","manches","mann","mehr","mein","meine","meinem","meinen","meiner","meines","mensch","menschen","mich","mir","mit","mittel","mochte","mochten","morgen","muss","musst","musste","mussten","muß","mußt","möchte","mögen","möglich","mögt","müssen","müsst","müßt","n","na","nach","nachdem","nahm","natürlich","neben","nein","neue","neuen","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nie","niemand","niemandem","niemanden","noch","nun","nur","o","ob","oben","oder","offen","oft","ohne","ordnung","p","q","r","recht","rechte","rechten","rechter","rechtes","richtig","rund","s","sa","sache","sagt","sagte","sah","satt","schlecht","schluss","schon","sechs","sechste","sechsten","sechster","sechstes","sehr","sei","seid","seien","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","selbst","sich","sie","sieben","siebente","siebenten","siebenter","siebentes","sind","so","solang","solche","solchem","solchen","solcher","solches","soll","sollen","sollst","sollt","sollte","sollten","sondern","sonst","soweit","sowie","später","startseite","statt","steht","suche","t","tag","tage","tagen","tat","teil","tel","tritt","trotzdem","tun","u","uhr","um","und","und?","uns","unse","unsem","unsen","unser","unsere","unserer","unses","unter","v","vergangenen","viel","viele","vielem","vielen","vielleicht","vier","vierte","vierten","vierter","viertes","vom","von","vor","w","wahr?","wann","war","waren","warst","wart","warum","was","weg","wegen","weil","weit","weiter","weitere","weiteren","weiteres","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wer","werde","werden","werdet","weshalb","wessen","wie","wieder","wieso","will","willst","wir","wird","wirklich","wirst","wissen","wo","woher","wohin","wohl","wollen","wollt","wollte","wollten","worden","wurde","wurden","während","währenddem","währenddessen","wäre","würde","würden","x","y","z","z.b","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zu","zuerst","zugleich","zum","zunächst","zur","zurück","zusammen","zwanzig","zwar","zwei","zweite","zweiten","zweiter","zweites","zwischen","zwölf","über","überhaupt","übrigens","niinkuin"],
    #     "DU":["aan","aangaande","aangezien","achte","achter","achterna","af","afgelopen","al","aldaar","aldus","alhoewel","alias","alle","allebei","alleen","alles","als","alsnog","altijd","altoos","ander","andere","anders","anderszins","beetje","behalve","behoudens","beide","beiden","ben","beneden","bent","bepaald","betreffende","bij","bijna","bijv","binnen","binnenin","blijkbaar","blijken","boven","bovenal","bovendien","bovengenoemd","bovenstaand","bovenvermeld","buiten","bv","daar","daardoor","daarheen","daarin","daarna","daarnet","daarom","daarop","daaruit","daarvanlangs","dan","dat","de","deden","deed","der","derde","derhalve","dertig","deze","dhr","die","dikwijls","dit","doch","doe","doen","doet","door","doorgaand","drie","duizend","dus","echter","een","eens","eer","eerdat","eerder","eerlang","eerst","eerste","eigen","eigenlijk","elk","elke","en","enig","enige","enigszins","enkel","er","erdoor","erg","ergens","etc","etcetera","even","eveneens","evenwel","gauw","ge","gedurende","geen","gehad","gekund","geleden","gelijk","gemoeten","gemogen","genoeg","geweest","gewoon","gewoonweg","haar","haarzelf","had","hadden","hare","heb","hebben","hebt","hedden","heeft","heel","hem","hemzelf","hen","het","hetzelfde","hier","hierbeneden","hierboven","hierin","hierna","hierom","hij","hijzelf","hoe","hoewel","honderd","hun","hunne","ieder","iedere","iedereen","iemand","iets","ik","ikzelf","in","inderdaad","inmiddels","intussen","inzake","is","ja","je","jezelf","jij","jijzelf","jou","jouw","jouwe","juist","jullie","kan","klaar","kon","konden","krachtens","kun","kunnen","kunt","laatst","later","liever","lijken","lijkt","maak","maakt","maakte","maakten","maar","mag","maken","me","meer","meest","meestal","men","met","mevr","mezelf","mij","mijn","mijnent","mijner","mijzelf","minder","miss","misschien","missen","mits","mocht","mochten","moest","moesten","moet","moeten","mogen","mr","mrs","mw","na","naar","nadat","nam","namelijk","nee","neem","negen","nemen","nergens","net","niemand","niet","niets","niks","noch","nochtans","nog","nogal","nooit","nu","nv","of","ofschoon","om","omdat","omhoog","omlaag","omstreeks","omtrent","omver","ondanks","onder","ondertussen","ongeveer","ons","onszelf","onze","onzeker","ooit","ook","op","opnieuw","opzij","over","overal","overeind","overige","overigens","paar","pas","per","precies","recent","redelijk","reeds","rond","rondom","samen","sedert","sinds","sindsdien","slechts","sommige","spoedig","steeds","tamelijk","te","tegen","tegenover","tenzij","terwijl","thans","tien","tiende","tijdens","tja","toch","toe","toen","toenmaals","toenmalig","tot","totdat","tussen","twee","tweede","u","uit","uitgezonderd","uw","vaak","vaakwat","van","vanaf","vandaan","vanuit","vanwege","veel","veeleer","veertig","verder","verscheidene","verschillende","vervolgens","via","vier","vierde","vijf","vijfde","vijftig","vol","volgend","volgens","voor","vooraf","vooral","vooralsnog","voorbij","voordat","voordezen","voordien","voorheen","voorop","voorts","vooruit","vrij","vroeg","waar","waarom","waarschijnlijk","wanneer","want","waren","was","wat","we","wederom","weer","weg","wegens","weinig","wel","weldra","welk","welke","werd","werden","werder","wezen","whatever","wie","wiens","wier","wij","wijzelf","wil","wilden","willen","word","worden","wordt","zal","ze","zei","zeker","zelf","zelfde","zelfs","zes","zeven","zich","zichzelf","zij","zijn","zijne","zijzelf","zo","zoals","zodat","zodra","zonder","zou","zouden","zowat","zulk","zulke","zullen","zult"],
    #     "FR":["a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd","hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu","un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tait","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"],
    #     "PT":["a","acerca","adeus","agora","ainda","alem","algmas","algo","algumas","alguns","ali","além","ambas","ambos","ano","anos","antes","ao","aonde","aos","apenas","apoio","apontar","apos","após","aquela","aquelas","aquele","aqueles","aqui","aquilo","as","assim","através","atrás","até","aí","baixo","bastante","bem","boa","boas","bom","bons","breve","cada","caminho","catorze","cedo","cento","certamente","certeza","cima","cinco","coisa","com","como","comprido","conhecido","conselho","contra","contudo","corrente","cuja","cujas","cujo","cujos","custa","cá","da","daquela","daquelas","daquele","daqueles","dar","das","de","debaixo","dela","delas","dele","deles","demais","dentro","depois","desde","desligado","dessa","dessas","desse","desses","desta","destas","deste","destes","deve","devem","deverá","dez","dezanove","dezasseis","dezassete","dezoito","dia","diante","direita","dispoe","dispoem","diversa","diversas","diversos","diz","dizem","dizer","do","dois","dos","doze","duas","durante","dá","dão","dúvida","e","ela","elas","ele","eles","em","embora","enquanto","entao","entre","então","era","eram","essa","essas","esse","esses","esta","estado","estamos","estar","estará","estas","estava","estavam","este","esteja","estejam","estejamos","estes","esteve","estive","estivemos","estiver","estivera","estiveram","estiverem","estivermos","estivesse","estivessem","estiveste","estivestes","estivéramos","estivéssemos","estou","está","estás","estávamos","estão","eu","exemplo","falta","fará","favor","faz","fazeis","fazem","fazemos","fazer","fazes","fazia","faço","fez","fim","final","foi","fomos","for","fora","foram","forem","forma","formos","fosse","fossem","foste","fostes","fui","fôramos","fôssemos","geral","grande","grandes","grupo","ha","haja","hajam","hajamos","havemos","havia","hei","hoje","hora","horas","houve","houvemos","houver","houvera","houveram","houverei","houverem","houveremos","houveria","houveriam","houvermos","houverá","houverão","houveríamos","houvesse","houvessem","houvéramos","houvéssemos","há","hão","iniciar","inicio","ir","irá","isso","ista","iste","isto","já","lado","lhe","lhes","ligado","local","logo","longe","lugar","lá","maior","maioria","maiorias","mais","mal","mas","me","mediante","meio","menor","menos","meses","mesma","mesmas","mesmo","mesmos","meu","meus","mil","minha","minhas","momento","muito","muitos","máximo","mês","na","nada","nao","naquela","naquelas","naquele","naqueles","nas","nem","nenhuma","nessa","nessas","nesse","nesses","nesta","nestas","neste","nestes","no","noite","nome","nos","nossa","nossas","nosso","nossos","nova","novas","nove","novo","novos","num","numa","numas","nunca","nuns","não","nível","nós","número","o","obra","obrigada","obrigado","oitava","oitavo","oito","onde","ontem","onze","os","ou","outra","outras","outro","outros","para","parece","parte","partir","paucas","pegar","pela","pelas","pelo","pelos","perante","perto","pessoas","pode","podem","poder","poderá","podia","pois","ponto","pontos","por","porque","porquê","portanto","posição","possivelmente","posso","possível","pouca","pouco","poucos","povo","primeira","primeiras","primeiro","primeiros","promeiro","propios","proprio","própria","próprias","próprio","próprios","próxima","próximas","próximo","próximos","puderam","pôde","põe","põem","quais","qual","qualquer","quando","quanto","quarta","quarto","quatro","que","quem","quer","quereis","querem","queremas","queres","quero","questão","quieto","quinta","quinto","quinze","quáis","quê","relação","sabe","sabem","saber","se","segunda","segundo","sei","seis","seja","sejam","sejamos","sem","sempre","sendo","ser","serei","seremos","seria","seriam","será","serão","seríamos","sete","seu","seus","sexta","sexto","sim","sistema","sob","sobre","sois","somente","somos","sou","sua","suas","são","sétima","sétimo","só","tal","talvez","tambem","também","tanta","tantas","tanto","tarde","te","tem","temos","tempo","tendes","tenha","tenham","tenhamos","tenho","tens","tentar","tentaram","tente","tentei","ter","terceira","terceiro","terei","teremos","teria","teriam","terá","terão","teríamos","teu","teus","teve","tinha","tinham","tipo","tive","tivemos","tiver","tivera","tiveram","tiverem","tivermos","tivesse","tivessem","tiveste","tivestes","tivéramos","tivéssemos","toda","todas","todo","todos","trabalhar","trabalho","treze","três","tu","tua","tuas","tudo","tão","tém","têm","tínhamos","um","uma","umas","uns","usa","usar","vai","vais","valor","veja","vem","vens","ver","verdade","verdadeiro","vez","vezes","viagem","vindo","vinte","você","vocês","vos","vossa","vossas","vosso","vossos","vários","vão","vêm","vós","zero","à","às","área","é","éramos","és","último"],
    #     "CA":["a","abans","ací","ah","així","això","al","aleshores","algun","alguna","algunes","alguns","alhora","allà","allí","allò","als","altra","altre","altres","amb","ambdues","ambdós","anar","ans","apa","aquell","aquella","aquelles","aquells","aquest","aquesta","aquestes","aquests","aquí","baix","bastant","bé","cada","cadascuna","cadascunes","cadascuns","cadascú","com","consegueixo","conseguim","conseguir","consigueix","consigueixen","consigueixes","contra","d","un","una","unes","uns","dalt","de","del","dels","des","des de","després","dins","dintre","donat","doncs","durant","e","eh","el","elles","ells","els","em","en","encara","ens","entre","era","erem","eren","eres","es","esta","estan","estat","estava","estaven","estem","esteu","estic","està","estàvem","estàveu","et","etc","ets","fa","faig","fan","fas","fem","fer","feu","fi","fins","fora","gairebé","ha","han","has","haver","havia","he","hem","heu","hi","ho","i","igual","iguals","inclòs","ja","jo","l","hi","la","les","li","li","n","llarg","llavors","m","he","ma","mal","malgrat","mateix","mateixa","mateixes","mateixos","me","mentre","meu","meus","meva","meves","mode","molt","molta","moltes","molts","mon","mons","més","n","hi","ne","ni","no","nogensmenys","només","nosaltres","nostra","nostre","nostres","o","oh","oi","on","pas","pel","pels","per","per que","perquè","però","poc","poca","pocs","podem","poden","poder","podeu","poques","potser","primer","propi","puc","qual","quals","quan","quant","que","quelcom","qui","quin","quina","quines","quins","què","s","ha","han","sa","sabem","saben","saber","sabeu","sap","saps","semblant","semblants","sense","ser","ses","seu","seus","seva","seves","si","sobre","sobretot","soc","solament","sols","som","son","sons","sota","sou","sóc","són","t","ha","han","he","ta","tal","també","tampoc","tan","tant","tanta","tantes","te","tene","tenim","tenir","teniu","teu","teus","teva","teves","tinc","ton","tons","tot","tota","totes","tots","un","una","unes","uns","us","va","vaig","vam","van","vas","veu","vosaltres","vostra","vostre","vostres","érem","éreu","és","éssent","últim","ús"]}
    
    nltk.download("stopwords")
    StopWords = {"EN": set(stopwords.words("english")),
        "FI": set(stopwords.words("finnish")),
        "GE": set(stopwords.words("german")),
        "DU": set(stopwords.words("dutch")),
        "FR": set(stopwords.words("french")),
        "PT": set(stopwords.words("portuguese")),
        "CA": ["a","abans","ací","ah","així","això","al","aleshores","algun","alguna","algunes","alguns","alhora","allà","allí","allò","als","altra","altre","altres","amb","ambdues","ambdós","anar","ans","apa","aquell","aquella","aquelles","aquells","aquest","aquesta","aquestes","aquests","aquí","baix","bastant","bé","cada","cadascuna","cadascunes","cadascuns","cadascú","com","consegueixo","conseguim","conseguir","consigueix","consigueixen","consigueixes","contra","d","un","una","unes","uns","dalt","de","del","dels","des","des de","després","dins","dintre","donat","doncs","durant","e","eh","el","elles","ells","els","em","en","encara","ens","entre","era","erem","eren","eres","es","esta","estan","estat","estava","estaven","estem","esteu","estic","està","estàvem","estàveu","et","etc","ets","fa","faig","fan","fas","fem","fer","feu","fi","fins","fora","gairebé","ha","han","has","haver","havia","he","hem","heu","hi","ho","i","igual","iguals","inclòs","ja","jo","l","hi","la","les","li","li","n","llarg","llavors","m","he","ma","mal","malgrat","mateix","mateixa","mateixes","mateixos","me","mentre","meu","meus","meva","meves","mode","molt","molta","moltes","molts","mon","mons","més","n","hi","ne","ni","no","nogensmenys","només","nosaltres","nostra","nostre","nostres","o","oh","oi","on","pas","pel","pels","per","per que","perquè","però","poc","poca","pocs","podem","poden","poder","podeu","poques","potser","primer","propi","puc","qual","quals","quan","quant","que","quelcom","qui","quin","quina","quines","quins","què","s","ha","han","sa","sabem","saben","saber","sabeu","sap","saps","semblant","semblants","sense","ser","ses","seu","seus","seva","seves","si","sobre","sobretot","soc","solament","sols","som","son","sons","sota","sou","sóc","són","t","ha","han","he","ta","tal","també","tampoc","tan","tant","tanta","tantes","te","tene","tenim","tenir","teniu","teu","teus","teva","teves","tinc","ton","tons","tot","tota","totes","tots","un","una","unes","uns","us","va","vaig","vam","van","vas","veu","vosaltres","vostra","vostre","vostres","érem","éreu","és","éssent","últim","ús"]}
    
    return


################################################################################ Results ###############################################################################

def elaborateStudy(books,k,fixedProb,logBase,n,out):
    global numWords, numTokens, results
    
    assert k>0, "Please define above zero the number of words to be presented."
    assert n>0, "Please define above zero the number of times the approximate counters are used."

    print("Running the exact counting of words in all books...")
    exactCounting(books,k,1)

    print("Running the approximate counting of words in all books " + str(n) + " times...")
    for i in range(1,n+1):
        approxCountingFixedProb(books,k,fixedProb,i)
        approxCountingLogarithmic(books,k,logBase,i)
    
    print("Calculating deviations and average of counts of the approximate counters' runs...")
    calculateDeviations(fixedProb,logBase,k,n)
    calculateApproximateCtrAverages(n)
    
    print("Generating information graphics...")
    generateCharts(k,fixedProb,logBase,out)

    print("Generating memory usage table...")
    generateMemoryUsageTable(k,out)
    
    print("Generating the results table...")
    generateResultsTable(k,out)
    
    print("All done!")
    return


def calculateDeviations(fixedProb,logBase,k,n):
    global results, deviations
    
    for counter in results.keys():
        if counter != "Exact Counter":
            for book in results[counter].keys():
                for language in results[counter][book].keys():

                    topWords = dict(results["Exact Counter"][book][language].most_common(k))

                    deviations[counter][book][language] = {}                    
                    deviations[counter][book][language]["maxdev"] = {}
                    deviations[counter][book][language]["mad"] = {}
                    deviations[counter][book][language]["mald"] = {}
                    deviations[counter][book][language]["stddev"] = {}
                    deviations[counter][book][language]["errPercent"] = {}
                    deviations[counter][book][language]["avgErrPercent"] = 0.0  # only top words considered

                    for word in results["Exact Counter"][book][language]:

                        deviations[counter][book][language]["maxdev"][word] = 0.0
                        deviations[counter][book][language]["mad"][word] = 0.0
                        deviations[counter][book][language]["stddev"][word] = 0.0

                        if counter == "Approximate Counter w/ Fixed Prob":
                            for run in results[counter][book][language]:
                                estimatedCount = results[counter][book][language][run][word] * (1/fixedProb)
                                curDev = abs(results["Exact Counter"][book][language][word] - estimatedCount)
                                if curDev > deviations[counter][book][language]["maxdev"][word]:
                                    deviations[counter][book][language]["maxdev"][word] = curDev
                                deviations[counter][book][language]["mad"][word] = deviations[counter][book][language]["mad"][word] + curDev
                                deviations[counter][book][language]["stddev"][word] = deviations[counter][book][language]["mad"][word] + curDev**2

                        if counter == "Approximate Counter w/ Logarithmic Prob":
                            deviations[counter][book][language]["mald"][word] = 0.0
                            for run in results[counter][book][language]:
                                estimatedCount = int((logBase**results[counter][book][language][run][word] - logBase + 1) / (logBase - 1))
                                curLogDev = abs(int(math.log(results["Exact Counter"][book][language][word]+1,logBase)) - results[counter][book][language][run][word])
                                curDev = abs(results["Exact Counter"][book][language][word] - estimatedCount)
                                if int(curDev) > deviations[counter][book][language]["maxdev"][word]:
                                    deviations[counter][book][language]["maxdev"][word] = curDev
                                deviations[counter][book][language]["mad"][word] = deviations[counter][book][language]["mad"][word] + curDev
                                deviations[counter][book][language]["mald"][word] = deviations[counter][book][language]["mald"][word] + curLogDev
                                deviations[counter][book][language]["stddev"][word] = deviations[counter][book][language]["mad"][word] + curDev**2
                            deviations[counter][book][language]["mald"][word] = deviations[counter][book][language]["mald"][word] / n
                        
                        deviations[counter][book][language]["mad"][word] = deviations[counter][book][language]["mad"][word] / n
                        deviations[counter][book][language]["stddev"][word] = math.sqrt(deviations[counter][book][language]["stddev"][word] / n)
                        deviations[counter][book][language]["errPercent"][word] = (deviations[counter][book][language]["mad"][word] * 100) / results["Exact Counter"][book][language][word]
                    
                    for top in topWords:
                        deviations[counter][book][language]["avgErrPercent"] = deviations[counter][book][language]["avgErrPercent"] + deviations[counter][book][language]["errPercent"][top]
                    deviations[counter][book][language]["avgErrPercent"] = deviations[counter][book][language]["avgErrPercent"] / k
             
    return


def calculateApproximateCtrAverages(n):
    global results

    for counter in results.keys():
        if counter != "Exact Counter":
            for book in results[counter].keys():
                for language in results[counter][book].keys():
                    resultsAvgRuns = Counter()
                    for run in results[counter][book][language]:
                        resultsAvgRuns.update(results[counter][book][language][run])
                    for key in resultsAvgRuns.keys():
                        resultsAvgRuns[key] = int(resultsAvgRuns[key]/n)
                    results[counter][book][language] = resultsAvgRuns
    
    return


def generateCharts(k,fixedProb,logBase,out):
    global results, deviations
    global titleAbbreviations, languageAbbreviations

    # Bar Charts

    # To study fidelity to reality amongst counters
    # for each book, for each language, each bar color is a counter, x = word, y = count
    
    index = np.arange(k)
    barWidth = 0.25
    barOpacity = 0.8
    for book in results["Exact Counter"].keys():                                                                       
        if len(results["Exact Counter"][book].keys()) == 0:
            continue
        for language in results["Exact Counter"][book].keys():
            topWords = dict(results["Exact Counter"][book][language].most_common(k))

            words = []
            exactCount = []
            approxCountFix = []
            approxCountLog = []

            for word in topWords:
                words.append(word)
                exactCount.append(results["Exact Counter"][book][language][word])
                approxCountFix.append(int(results["Approximate Counter w/ Fixed Prob"][book][language][word] * (1/fixedProb)))
                approxCountLog.append(int((logBase**results["Approximate Counter w/ Logarithmic Prob"][book][language][word] - logBase + 1) / (logBase - 1)))

            fig, ax = plt.subplots()
            chartTitle = "Word Counts of '" + titleAbbreviations[book] + "' in " + languageAbbreviations[language]

            rects1 = plt.bar(index, exactCount, barWidth, alpha=barOpacity, color='b', label='Exact Ctr')
            rects2 = plt.bar(index + barWidth, approxCountFix, barWidth, alpha=barOpacity, color='y', label='Fixed Probability Ctr (P=0.5)')
            rects3 = plt.bar(index + barWidth*2, approxCountLog, barWidth, alpha=barOpacity, color='g', label='Logarithmic Probability Ctr (Log Base = sqrt(2))')

            plt.xlabel('Words')
            plt.ylabel('Count')
            plt.title(chartTitle)
            plt.xticks(index + barWidth, words)
            plt.legend()

            #plt.tight_layout()
            #plt.show()
            plt.savefig(out + "charts/wordCount_" + book + "_" + language + ".png")
            plt.close()
    
    # Scatter Plots

    colors = ["red", "green", "blue","yellow","purple"]
    
    # To study deviations between languages:
    # for each book, for each approx counter, each color is a language, x = count, y = deviation
    '''
    for book in results["Exact Counter"].keys():                                                                       
        if len(results["Exact Counter"][book].keys()) == 0:
            continue

        for counter in results.keys():
            if counter == "Exact Counter":
                continue

            groups = []
            languages = []
            
            for language in results["Exact Counter"][book].keys():
                
                topWords = dict(results["Exact Counter"][book][language].most_common(k))
                approxCount = []
                dev = []

                for word in topWords: 
                    approxCount.append(results[counter][book][language][word])
                    dev.append(deviations[counter][book][language]["mad"][word])

                groups.append((approxCount,dev))
                languages.append(language)

            fig, ax = plt.subplots()
            plotTitle = "Deviations of Counts for '" + titleAbbreviations[book] + "'\n in All Languages"

            for gr, co, la in zip(groups, colors[:len(groups)], languages):
                x, y = gr
                ax.scatter(x,y, alpha=0.8, c=co, edgecolors='none', s=30, label=languageAbbreviations[la])

            plt.xlabel("Counts of " + counter)
            plt.ylabel("Deviation")
            plt.title(plotTitle)
            plt.legend()
            if counter == "Approximate Counter w/ Fixed Prob":
                plt.savefig(out + "plots/languageComparison_ApproxFP_" + book + ".png")
            elif counter == "Approximate Counter w/ Logarithmic Prob":
                plt.savefig(out + "plots/languageComparison_ApproxLP_" + book + ".png")
            plt.close()
    '''
    # To study deviations between counters:
    # for each book, for each language, each color is an approx counter, x = deviation on order, y = deviation on count
    
    for book in results["Exact Counter"].keys():                                                                       
        if len(results["Exact Counter"][book].keys()) == 0:
            continue
            
        for language in results["Exact Counter"][book].keys():

            trueTopWords = results["Exact Counter"][book][language].most_common(k)
            groups = []
            counters = []

            for counter in results.keys():
                if counter == "Exact Counter":
                    continue

                topWords = results[counter][book][language].most_common()
                devX = []
                for i,tTW in enumerate(trueTopWords):
                    contains = False
                    for j,tW in enumerate(topWords):
                        if tTW[0] == tW[0]:
                            devX.append(abs(i-j))
                            contains = True
                            break
                    assert contains, "Something went wrong during the counting phase of '" + counter + "'."

                devY = []
                if counter == "Approximate Counter w/ Fixed Prob":
                    for tTW in trueTopWords:
                        devY.append(deviations[counter][book][language]["mad"][tTW[0]])
                elif counter == "Approximate Counter w/ Logarithmic Prob":
                    for tTW in trueTopWords:
                        devY.append(deviations[counter][book][language]["mad"][tTW[0]])

                groups.append((devX,devY))
                counters.append(counter)

            fig, ax = plt.subplots()
            plotTitle = "Deviations of Count and Order for\n'" + titleAbbreviations[book] + "' in " + languageAbbreviations[language]

            for gr, co, ctr in zip(groups, colors[:len(groups)], counters):
                x, y = gr
                ax.scatter(x,y, alpha=0.8, c=co, edgecolors='none', s=30, label=ctr)

            plt.xlabel("Order Deviation")
            plt.ylabel("Counter Mean Average Deviation")
            plt.title(plotTitle)
            plt.legend() # (loc=2)
            plt.savefig(out + "plots/counterComparison_" + book + "_" + language + ".png")
            plt.close()
            
    return


def generateMemoryUsageTable(k,out):
    global results
    global numWords, numTokens

    csvFile = open(out + "memory_usage.csv","w", encoding="utf-8")

    # Total number of words and of relevant words only of each book in each translation
    csvFile.write("Total number of words and of relevant words only of each book in each translation:\n")
    csvFile.write("{:s},{:s},{:s},{:s}\n".format("Book","Language","All Words","Relevant Words"))

    for book in numWords.keys():
        fstLineBook = True
        for language in numWords[book].keys():
            fstLineLang = True
            if fstLineBook:
                csvFile.write("{:s},{:s},{:d},{:d}\n".format(book,language,numWords[book][language],numTokens[book][language]))
                fstLineBook = False
                fstLineLang = False
            else:
                if fstLineLang:
                    csvFile.write("{:s},{:s},{:d},{:d}\n".format("",language,numWords[book][language],numTokens[book][language]))
                    fstLineLang = False
                else:
                    csvFile.write("{:s},{:s},{:d},{:d}\n".format("","",numWords[book][language],numTokens[book][language]))

    # Total number of bytes used by the values on each individual counter of each solution
    csvFile.write("\n")
    csvFile.write("Total number of bytes used by the values on each individual counter of each solution:\n")

    EC_countsSize = 0
    ACFP_countsSize = 0
    ACLP_countsSize = 0

    EC_countsSizeTop = 0
    ACFP_countsSizeTop = 0
    ACLP_countsSizeTop = 0

    for book in results["Exact Counter"].keys():
        for language in results["Exact Counter"][book].keys():
            for word in results["Exact Counter"][book][language].keys():
                EC_countsSize += getsizeof(bytes(results["Exact Counter"][book][language][word]))

            topWords = results["Exact Counter"][book][language].most_common(k)
            for word in topWords:
                EC_countsSizeTop += getsizeof(bytes(word[1]))

    for book in results["Approximate Counter w/ Fixed Prob"].keys():
        for language in results["Approximate Counter w/ Fixed Prob"][book].keys():
            for word in results["Approximate Counter w/ Fixed Prob"][book][language].keys():
                ACFP_countsSize += getsizeof(bytes(results["Approximate Counter w/ Fixed Prob"][book][language][word]))

            topWords = results["Approximate Counter w/ Fixed Prob"][book][language].most_common(k)
            for word in topWords:
                ACFP_countsSizeTop += getsizeof(bytes(word[1]))

    for book in results["Approximate Counter w/ Logarithmic Prob"].keys():
        for language in results["Approximate Counter w/ Logarithmic Prob"][book].keys():
            for word in results["Approximate Counter w/ Logarithmic Prob"][book][language].keys():
                ACLP_countsSize += getsizeof(bytes(results["Approximate Counter w/ Logarithmic Prob"][book][language][word]))

            topWords = results["Approximate Counter w/ Logarithmic Prob"][book][language].most_common(k)
            for word in topWords:
                ACLP_countsSizeTop += getsizeof(bytes(word[1]))

    csvFile.write("{:s},{:s},{:s},{:s}\n".format("","EC","ACFP","ACLP"))
    csvFile.write("{:s},{:d},{:d},{:d}\n".format("All Words",EC_countsSize,ACFP_countsSize,ACLP_countsSize))
    csvFile.write("{:s},{:d},{:d},{:d}\n".format("Top " + str(k) + " Words",EC_countsSizeTop,ACFP_countsSizeTop,ACLP_countsSizeTop))

    csvFile.close()
    return


def generateResultsTable(k,out):
    global results, deviations

    for counter in results.keys():
        for book in results[counter].keys():
            for language in results[counter][book].keys():
                results[counter][book][language] = results[counter][book][language].most_common()#.most_common(k)

    line = "+------++-------------+-----------++-------------+--------------+--------------+---------+---------+------------------+-------------++-------------+--------------+------------------+--------------+---------+---------+------------------+-------------"
    AF = "Approximate Counter w/ Fixed Prob"
    AL = "Approximate Counter w/ Logarithmic Prob"

    txtFile = open(out + "results.txt","w", encoding="utf-8")
    csvFile = open(out + "results.csv","w", encoding="utf-8")

    #print(" {:4s} | {:4s} || {:11s} | {:9s} || {:11s} | {:12s} | {:12s} | {:7s} | {:7s} | {:16s} | {:11s} || {:11s} | {:12s} | {:16s} | {:12s} | {:7s} | {:7s} | {:16s} | {:11s} \n".format("Book","Lang","Words","Exact Ctr","Words","Fix Prob Ctr","Mean Avg Dev","Std Dev","Max Dev","Mean Rel Err (%)","Avg Err (%)","Words","Log Prob Ctr","Mean Avg Log Dev","Mean Avg Dev","Std Dev","Max Dev","Mean Rel Err (%)","Avg Err (%)"))
    #print("------" + line)

    txtFile.write(" {:4s} | {:4s} || {:11s} | {:9s} || {:11s} | {:12s} | {:12s} | {:7s} | {:7s} | {:16s} | {:11s} || {:11s} | {:12s} | {:16s} | {:12s} | {:7s} | {:7s} | {:16s} | {:11s} \n".format("Book","Lang","Words","Exact Ctr","Words","Fix Prob Ctr","Mean Avg Dev","Std Dev","Max Dev","Mean Rel Err (%)","Avg Err (%)","Words","Log Prob Ctr","Mean Avg Log Dev","Mean Avg Dev","Std Dev","Max Dev","Mean Rel Err (%)","Avg Err (%)"))
    txtFile.write("------" + line + "\n")
    csvFile.write("{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s},{:s}\n".format("Book","Lang","Words","Exact Ctr","Words","Fix Prob Ctr","Mean Avg Dev","Std Dev","Max Dev","Mean Rel Err (%)","Avg Err (%)","Words","Log Prob Ctr","Mean Avg Log Dev","Mean Avg Dev","Std Dev","Max Dev","Mean Rel Err (%)","Avg Err (%)"))
    
    for book in results["Exact Counter"].keys():                                                                       
        if len(results["Exact Counter"][book].keys()) == 0:
            continue

        #print("------" + line)
        txtFile.write("------" + line + "\n")
        first = book

        for language in results["Exact Counter"][book].keys():
            wordE = results["Exact Counter"][book][language][0][0]
            countE = results["Exact Counter"][book][language][0][1]
            
            wordAF = results[AF][book][language][0][0]
            countAF = results[AF][book][language][0][1]
            maDevAF = deviations[AF][book][language]["mad"][wordAF]
            stdDevAF = deviations[AF][book][language]["stddev"][wordAF]
            maxDevAF = deviations[AF][book][language]["maxdev"][wordAF]
            meanRelErrAF = deviations[AF][book][language]["errPercent"][wordAF]
            avgErrAF = deviations[AF][book][language]["avgErrPercent"] 

            wordAL = results[AL][book][language][0][0]
            countAL = results[AL][book][language][0][1]
            maLogDevAL = deviations[AL][book][language]["mald"][wordAF]
            maDevAL = deviations[AL][book][language]["mad"][wordAF]
            stdDevAL = deviations[AL][book][language]["stddev"][wordAF]
            maxDevAL = deviations[AL][book][language]["maxdev"][wordAF]
            meanRelErrAL = deviations[AL][book][language]["errPercent"][wordAF]
            avgErrAL = deviations[AL][book][language]["avgErrPercent"] 

            #print(" {:4s} | {:4s} || {:11s} | {:9d} || {:11s} | {:12d} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11.2f} || {:11s} | {:12d} | {:16.2f} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11.2f} ".format(first, language, wordE, countE, wordAF, countAF, maDevAF, stdDevAF, maxDevAF, meanRelErrAF, avgErrAF, wordAL, countAL, maLogDevAL, maDevAL, stdDevAL, maxDevAL, meanRelErrAL, avgErrALL))
            txtFile.write(" {:4s} | {:4s} || {:11s} | {:9d} || {:11s} | {:12d} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11.2f} || {:11s} | {:12d} | {:16.2f} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11.2f} \n".format(first, language, wordE, countE, wordAF, countAF, maDevAF, stdDevAF, maxDevAF, meanRelErrAF, avgErrAF, wordAL, countAL, maLogDevAL, maDevAL, stdDevAL, maxDevAL, meanRelErrAL, avgErrAL))
            csvFile.write("{:s},{:s},{:s},{:d},{:s},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:s},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(first, language, wordE, countE, wordAF, countAF, maDevAF, stdDevAF, maxDevAF, meanRelErrAF, avgErrAF, wordAL, countAL, maLogDevAL, maDevAL, stdDevAL, maxDevAL, meanRelErrAL, avgErrAL))
            
            for i in range(1,k):
                wordE = results["Exact Counter"][book][language][i][0]
                countE = results["Exact Counter"][book][language][i][1]

                wordAF = results[AF][book][language][i][0]
                countAF = results[AF][book][language][i][1]
                maDevAF = deviations[AF][book][language]["mad"][wordAF]
                stdDevAF = deviations[AF][book][language]["stddev"][wordAF]
                maxDevAF = deviations[AF][book][language]["maxdev"][wordAF]
                meanRelErrAF = deviations[AF][book][language]["errPercent"][wordAF]

                wordAL = results[AL][book][language][i][0]
                countAL = results[AL][book][language][i][1]
                maLogDevAL = deviations[AL][book][language]["mald"][wordAF]
                maDevAL = deviations[AL][book][language]["mad"][wordAF]
                stdDevAL = deviations[AL][book][language]["stddev"][wordAF]
                maxDevAL = deviations[AL][book][language]["maxdev"][wordAF]
                meanRelErrAL = deviations[AL][book][language]["errPercent"][wordAF]

                #print(" {:4s} | {:4s} || {:11s} | {:9d} || {:11s} | {:12d} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11s} || {:11s} | {:12d} | {:16.2f} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11s} ".format("", "", wordE, countE, wordAF, countAF, maDevAF, stdDevAF, maxDevAF, meanRelErrAF, "", wordAL, countAL, maLogDevAL, maDevAL, stdDevAL, maxDevAL, meanRelErrAL, ""))
                txtFile.write(" {:4s} | {:4s} || {:11s} | {:9d} || {:11s} | {:12d} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11s} || {:11s} | {:12d} | {:16.2f} | {:12.2f} | {:7.2f} | {:7.2f} | {:16.2f} | {:11s} \n".format("", "", wordE, countE, wordAF, countAF, maDevAF, stdDevAF, maxDevAF, meanRelErrAF, "", wordAL, countAL, maLogDevAL, maDevAL, stdDevAL, maxDevAL, meanRelErrAL, ""))
                csvFile.write("{:s},{:s},{:s},{:d},{:s},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:s},{:s},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:s}\n".format("", "", wordE, countE, wordAF, countAF, maDevAF, stdDevAF, maxDevAF, meanRelErrAF, "", wordAL, countAL, maLogDevAL, maDevAL, stdDevAL, maxDevAL, meanRelErrAL, ""))
                                                                                                                                                    
            first = ""
            
            if language != list(results["Exact Counter"][book].keys())[-1]:
                #print("      " + line)
                txtFile.write("      " + line + "\n")

        #print("------" + line)
        txtFile.write("------" + line + "\n")

    #print("------" + line)
    txtFile.write("------" + line + "\n")

    txtFile.close()
    csvFile.close()

    return

prepareGlobalVariables()
elaborateStudy(Books,K,FixedProb,LogBase,N,Out)

#exactCounting(Books,K)
#approxCountingFixedProb(Books,K,FixedProb)
#approxCountingLogarithmic(Books,K,LogBase)

#print(results)
#print(results["Approximate Counter w/ Fixed Prob"]["ACC"]["EN"])
#print(results["Approximate Counter w/ Logarithmic Prob"]["OT"]["EN"])

