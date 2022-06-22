from itertools import combinations
from pprint import pprint


# span l1 set
# use l1 set to make c2 set, discard all non-l1 sets and make l2 set
# use l2 set to make c3 set, discard all non-l2 sets and make l3 set
# use l3 set to make c4 set, discard all non-l3 sets and make l4 set

class AprioriAlg:
    def __init__(self) -> None:
        self.supportThreshold = 19
        self.allFavoriteBooks = self.readBooksIntoSet()
        self.oneItems = self.getOneItems()
        self.L1Items = self.getL1Support()
        self.allFrequentSets = []

    def readBooksIntoSet(self) -> list:
        with open("books.txt", 'r') as f:
            books = [set(line.rstrip('\n').split(';')) for line in f]
        return books

    def writeL1Books(self):
        with open('oneItems.txt', 'w') as f:
            for k, v in self.L1Items.items():
                f.write(f"{v}:{list(k)[0]}\n")
        f.close()

    def writeAllBooks(self):
        with open('patterns.txt', 'w') as f:
            for k, v in self.runApriori().items(): # k = itemset size, v = dict of frozensets
                for bookSet, support in v.items(): # k = set of books that is frequent, support of set
                    lineString = ';'.join(list(bookSet))
                    f.write(f"{support}:{lineString}\n")
        f.close()

    def getOneItems(self):
        oneItems = set()
        for x in self.allFavoriteBooks:
            for book in x:
                oneItems.add(frozenset([book])) # use frozenset because we only want 1 single item in set
        return oneItems

    def getL1Support(self) -> dict:
        tempDict = {}
        for book in self.oneItems:
            support = 0
            for favoriteBookSet in self.allFavoriteBooks:
                if book.issubset(favoriteBookSet):
                    support = support + 1
                tempDict[book] = support
        return {book:tempDict.get(book) for book in tempDict if tempDict.get(book) >= self.supportThreshold}

    def runApriori(self):
        itemSetToCheck = {} # dict not set because need to keep support
        candidates = self.oneItems # already calculated on instantiation, eh egal
        lStepItems = self.L1Items # already calculated on instantiation
        step = 2 # start with 2 because l1 items are just copied
        while not len(lStepItems) == 0:
            # get latest frequent itemset
            itemSetToCheck[step - 1] = lStepItems
            # source: slides | build new candidate set
            candidates = self.selfJoinCandiates(lStepItems, step)
            
            # prune infrequent itemsets NOT NEEDED BECAUSE ONLY APPEND FREQUENT SETS
            # candidates = self.prune(candidates, lStepItems, step)
            
            # get frequent itemsets
            lStepItems = self.getFrequentItemsets(candidates)
            
            # append set to all frequent sets
            if not len(lStepItems) == 0:
                self.allFrequentSets.append(lStepItems)
            
            # increase step, dont forget this lol
            step = step + 1
        
        return itemSetToCheck
            
    def selfJoinCandiates(self, candidates, step) -> dict:
        # source: https://stackoverflow.com/questions/2541401/pairwise-crossproduct-in-python
        return {x.union(y) for x in candidates for y in candidates if len(x.union(y)) == step}

    def prune(self, candidates, lStepItems, step):
        tempCandidates = set()
        for x in candidates:
            for combination in combinations(x, step):
                if not combination in lStepItems:
                    tempCandidates.add(frozenset(combination))
        return tempCandidates

    def getFrequentItemsets(self, candidates):
        tempDict = {}
        for itemset in candidates:
            support = 0
            for favoriteBookSet in self.allFavoriteBooks:
                if itemset.issubset(favoriteBookSet):
                    support = support + 1
                tempDict[itemset] = support        
        return {book:tempDict.get(book) for book in tempDict if tempDict.get(book) >= self.supportThreshold}

    def getConfidence(self, itemSetToCheck):
        lenCheck = len(itemSetToCheck) + 1
        indexOfPlus1Sets = lenCheck - 2
        highestSupport = 0
        recommendedItemset = ""
        for key, value in self.allFrequentSets[indexOfPlus1Sets].items():
            # key is set
            # value is confidence            
            lenCheck = len(itemSetToCheck) + 1
            
            if len(key) == lenCheck:
                if itemSetToCheck.issubset(key):
                    if value > highestSupport:
                        highestSupport = value
                        recommendedItemset = key
        
        print (f"Recommendation: {next(iter(recommendedItemset.difference(itemSetToCheck)))} w/ a support of: {highestSupport}")



def main():
    apriori = AprioriAlg()
    
    apriori.runApriori()

    # part 1
    apriori.writeL1Books()

    # part 2
    apriori.writeAllBooks()

    # part 3
    testSet = frozenset(["Harry Potter and the Sorcerers Stone (Book 1)", "Harry Potter and the Chamber of Secrets (Book 2)"])
    apriori.getConfidence(testSet)
    

if __name__ == "__main__":    
    main()

