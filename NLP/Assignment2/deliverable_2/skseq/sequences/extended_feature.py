from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures

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

        # Suffixes
        # max_suffix = 3
        # for i in range(max_suffix):
        #     if len(word) > i+1:
        #         suffix = word[-(i+1):]
        #         # Generate feature name.
        #         feat_name = "suffix:%s::%s" % (suffix, y_name)
        #         # Get feature ID from name.
        #         feat_id = self.add_feature(feat_name)
        #         # Append feature.
        #         if feat_id != -1:
        #             features.append(feat_id)

        #First letter is capitalized
        if word[0].isupper():
            # Generate feature name.
            feat_name = "capi_ini::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Any other letter is capitalized      
        if len(word)>1 and not str.islower(word[1:]):
            # Generate feature name.
            feat_name = "capi_any::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)


        #We encounter a digit
        if str.isdigit(word):
            # Generate feature name.
            feat_name = "digit::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Word contains a digit
        if any(char.isdigit() for char in word) and not str.isdigit(word):
            # Generate feature name.
            feat_name = "insidedigit::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)


        #The word contains the char "point" (U.S.A)
        if str.find(word, ".") != -1:
            # Generate feature name.
            feat_name = "inside_point::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Explicit suffixes
        #ING     'during their trip'
        if str.endswith(word,'ing'):
            # Generate feature name.
            feat_name = "ending_ing::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
        #ED
        if str.endswith(word,'ed'):
            # Generate feature name.
            feat_name = "ending_ed::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
        #NESS
        if str.endswith(word,'ness'):
            # Generate feature name.
            feat_name = "ending_ness::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
        #SHIP
        if str.endswith(word,'ship'):
            # Generate feature name.
            feat_name = "ending_ship::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
        #ITY
        if str.endswith(word,'ity'):
            # Generate feature name.
            feat_name = "ending_ity::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #TY
        if str.endswith(word,'ty'):
            # Generate feature name.
            feat_name = "ending_ty::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #LY
        if str.endswith(word,'ly'):
            # Generate feature name.
            feat_name = "ending_ly::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Word contains a hypen
        if str.find(word, "-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Preposition 'to', so "Jack London went to Paris." helps identifying Paris
        if word == 'to':
            # Generate feature name.
            feat_name = "prep_to::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Preposition 'of', so "The president of the United States of America" helps identifying America
        if word == 'of':
            # Generate feature name.
            feat_name = "prep_of::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Preposition 'from', so "from/O Barcelona/B-org" is corrected
        if word == 'from':
            # Generate feature name.
            feat_name = "prep_from::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Article 'the', lots of names and other tokens are followed by 'the'
        if word == 'the':
            # Generate feature name.
            feat_name = "article_the::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)


        return features


class ExtendedUnicodeFeatures(UnicodeFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = y

        # Get word name from ID.
        x_name = x

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        feat_name = str(feat_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if str.istitle(word):
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.isdigit(word):
            # Generate feature name.
            feat_name = "number::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.find(word, "-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features
