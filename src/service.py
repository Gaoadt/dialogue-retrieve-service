from src.embeddings import *
from src.clustering import *
from src.graph import FrequencyDialogueGraph
from src.dataset import load_dataset
from tqdm import trange

class RetrieveService:
    def __init__(self, train, embedder, embeddings, cluster_count, response_emb):
        self.algo = MiniBatchKMeansClustering
        self.embeddings = embeddings
        self.embedder = embedder
        self.cluster_count = cluster_count
        self.clustering = self.algo(n_clusters=cluster_count).fit(self.embeddings)
        self.clustering.clustering.labels_ = self.clustering.clustering.predict(self.embedder.embeddings)
        self.graph = FrequencyDialogueGraph(train, self.embedder, self.clustering)
        self.graph._build()
        self.train = train
        self.response_emb = response_emb

        self.db = [[] for i in range(cluster_count)]
        for i in trange((len(response_emb))):
            self.db[self.clustering.clustering.predict(np.array([response_emb[i]]))[0]].append(i)

    def next_cluster(self, context_embedding):
        arg = np.array([context_embedding.tolist()], dtype=np.float32)
        current_cluster = self.clustering.clustering.predict(arg)[0]
        return self.graph.probabilities[current_cluster].argmax()

    def retrieve(self, context_embedding):
        next = self.next_cluster(context_embedding)
        emb = self.response_emb[self.db[next]]
        return self.train.utterances[self.db[next][context_embedding.dot(emb.T).argmax()]]


def build_service(cluster_count=100):

    test, val, train = load_dataset()

    embeddings_dir = Path("data/multiwoz_convert_embeddings")
    test_convert = np.load(embeddings_dir / "test_convert_context.np.npy")
    resp_test_convert = np.load(embeddings_dir / "test_convert_responses.np.npy")

    train_convert = np.load(embeddings_dir / "train_convert_context.np.npy")
    resp_train_convert = np.load(embeddings_dir / "train_convert_responses.np.npy")

    cached_convert_context = CachedEmbeddings(train, train_convert, test, test_convert)
    cached_convert_response = CachedEmbeddings(train, resp_train_convert, test, resp_test_convert)

    service = RetrieveService(train, cached_convert_context, train_convert, cluster_count, resp_train_convert)
    return service