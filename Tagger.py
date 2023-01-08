class Tagger:
    @staticmethod
    def __tag_to_role_of_speech(tag):
        if tag.startswith('N'):
            return 'n'
        if tag.startswith('V'):
            return 'v'
        if tag.startswith('J'):
            return 'a'
        if tag.startswith('R'):
            return 'r'
        return None

    @staticmethod
    def tags_to_roles_of_speech(tags):
        roles = []
        for tag in tags:
            roles.append(Tagger.__tag_to_role_of_speech(tag[1]))
        return roles

