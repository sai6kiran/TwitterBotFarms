from nltk.corpus import wordnet as wn
import spacy

def convert(word, from_pos, to_pos):

	# Just to make it a bit more readable
	WN_NOUN = 'n'
	WN_VERB = 'v'
	WN_ADJECTIVE = 'a'
	WN_ADJECTIVE_SATELLITE = 's'
	WN_ADVERB = 'r'

	synsets = wn.synsets(word, pos=from_pos)

	# Word not found
	if not synsets:
		return []

	# Get all lemmas of the word (consider 'a'and 's' equivalent)
	lemmas = []
	for s in synsets:
		for l in s.lemmas():
			if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
				lemmas += [l]

	# Get related forms
	derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

	# filter only the desired pos (consider 'a' and 's' equivalent)
	related_noun_lemmas = []

	for drf in derivationally_related_forms:
		for l in drf[1]:
			if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
				related_noun_lemmas += [l]

	# Extract the words from the lemmas
	words = [l.name() for l in related_noun_lemmas]
	len_words = len(words)

	# Build the result in the form of a list containing tuples (word, probability)
	result = [(w, float(words.count(w)) / len_words) for w in set(words)]
	result.sort(key=lambda w:-w[1])

	# return all the possibilities sorted by probability
	return result

def ObtainStemAndLemmatizationWord(stp):
	if(wn.synsets(stp)):	#Is the word recognizable in the NLP Python Library to get its form.
		low = convert(stp, wn.synsets(stp)[0].pos(), "n")	#stp: String to parse into its noun form

		lnw = 1000	#length of word
		wrd=" "		#Word returned
		prw = 0		#Probability this is the root word
		for i in low:
        		if(len(i[0])<lnw and i[1]>=prw):
                		wrd = i[0]
                		lnw = len(i[0])
                		prw = i[1]

		nlp = spacy.load('en', disable=['parser', 'ner'])
		# Parse
		doc = nlp(stp)

		# Extract the lemma
		orw = " "	#Original Root Word
		for token in doc:
        		orw = str(token.lemma_)

		return (wrd, orw)
	else:
		return (stp, None)

#print(convert("RT", wn.synsets("RT")[0].pos(), "n"))
