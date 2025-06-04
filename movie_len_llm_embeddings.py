from embedding.llm_embedder import LLMEmbedder


if __name__ == "__main__":
    embedder = LLMEmbedder("Qwen/Qwen3-1.7B")
    # Example movie descriptions from IMDb
    movie_descriptions = [
        "A young boy wizard discovers his magical heritage on his eleventh birthday when he receives a letter of acceptance to Hogwarts School of Witchcraft and Wizardry.",
        "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "A thief who steals information by entering people's dreams takes one last job: planting an idea into a target's subconscious.",
        "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."
    ]
    embeddings = embedder.embed(movie_descriptions)
    print(embeddings)