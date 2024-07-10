from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures
from skseq.sequences.word_lists import prepositions, stopwords, chem_prefix, SI_prefixes, honorifics, person_suffix, corps_abbr, geographical_suffix

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        
        # titlecased
        if str.istitle(word):
            # generate  feature name
            feat_name = "titlecased::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)

        # uppercased
        if str.isupper(word):
            # generate  feature name
            feat_name = "uppercased::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)

        # lowercased
        if str.islower(word):
            # generate  feature name
            feat_name = "lowercased::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # is a digit
        if str.isdigit(word):
            # generate  feature name
            feat_name = "isdigit::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # the word has a number
        if any(char.isdigit() for char in word):
            # generate  feature name
            feat_name = "hasnumber::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # the word is alpha numeric
        if str.isalnum(word):
            # generate  feature name
            feat_name = "alphanum::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # the word is alphabetic characters only
        if str.isalpha(word):
            # generate  feature name
            feat_name = "alpha::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # days of the week
        if str.endswith(word, "day"):
            # generate  feature name
            feat_name = "weekday::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # some months (mostly)
        if str.endswith(word, "ber"):
            # generate  feature name
            feat_name = "monthber::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # detect ing suffix
        if str.endswith(word, "ing"):
            # generate  feature name
            feat_name = "suffixing::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # word is a preposition
        if word.lower() in prepositions:
            # generate  feature name
            feat_name = "preposition::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id =self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # word is a stopword
        if word.lower() in stopwords:
            # generate  feature name
            feat_name = "stopword::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id = self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)

        # word has a hyphen
        if "-" in word:
            # generate  feature name
            feat_name = "hyphened::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id = self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # word has a dot
        if "." in word:
            # generate  feature name
            feat_name = "dotted::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id = self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # word has a single quote
        if "'" in word:
            # generate  feature name
            feat_name = "quoted::%s" % y_name
            feat_name = str(feat_name)

            #get feature id from name
            feat_id = self.add_feature(feat_name)
            # append feature
            if feat_id != -1:
                features.append(feat_id)
        
        # word has a chemical prefix
        for pre in chem_prefix:
            if str.startswith(word.lower(), pre):
                # generate  feature name
                feat_name = f"chempre{pre}::{y_name}"
                feat_name = str(feat_name)

                #get feature id from name
                feat_id = self.add_feature(feat_name)
                # append feature
                if feat_id != -1:
                    features.append(feat_id)
        
        # word has a SI prefix
        for pre in SI_prefixes:
            if str.startswith(word.lower(), pre):
                # generate  feature name
                feat_name = f"siprefix{pre}::{y_name}"
                feat_name = str(feat_name)

                #get feature id from name
                feat_id = self.add_feature(feat_name)
                # append feature
                if feat_id != -1:
                    features.append(feat_id)
        
        # word has an honorific
        for pre in honorifics:
            if str.startswith(word.lower(), pre):
                # generate  feature name
                feat_name = f"honorifics{pre}::{y_name}"
                feat_name = str(feat_name)

                #get feature id from name
                feat_id = self.add_feature(feat_name)
                # append feature
                if feat_id != -1:
                    features.append(feat_id)
        
        # word has a person suffix
        for suf in person_suffix:
            if str.endswith(word.lower(), suf):
                # generate  feature name
                feat_name = f"personsuffix{suf}::{y_name}"
                feat_name = str(feat_name)

                #get feature id from name
                feat_id = self.add_feature(feat_name)
                # append feature
                if feat_id != -1:
                    features.append(feat_id)
        
        # word has a person suffix
        for suf in geographical_suffix:
            if str.endswith(word.lower(), suf):
                # generate  feature name
                feat_name = f"geographsuffix{suf}::{y_name}"
                feat_name = str(feat_name)

                #get feature id from name
                feat_id = self.add_feature(feat_name)
                # append feature
                if feat_id != -1:
                    features.append(feat_id)
        
        # word has a corporate abbreviation
        for abbr in corps_abbr:
            if str.startswith(word.lower(), abbr) or str.endswith(word.lower(), abbr):
                # generate  feature name
                feat_name = f"corpabbr{abbr}::{y_name}"
                feat_name = str(feat_name)

                #get feature id from name
                feat_id = self.add_feature(feat_name)
                # append feature
                if feat_id != -1:
                    features.append(feat_id)
        return features
    

