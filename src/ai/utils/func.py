from random_word import RandomWords


def random_run_name(num_words=3):
    generator = RandomWords()
    return "-".join([generator.get_random_word().lower() for _ in range(num_words)])
