# Imports
import hashlib
import time


class Shinglings:
    """
    A class for creating k-shingles (substrings of length k) from a document.
    """
    def __init__(self, k):
        """
        Initialize the Shinglings class with the length of each shingle.
        """
        self.k = k

    def create_shingles(self, document):
        """
        Generate a set of k-shingles from the input document.
        """
        shingles = set()
        for i in range(len(document) - self.k + 1):
            shingle = document[i:i + self.k]
            shingles.add(shingle)
        return shingles


class MinHashing:
    """
    A class for generating MinHash signatures for sets of shingles.
    """
    def __init__(self, num_hashes):
        """
        Initialize the MinHashing class with the number of hash functions.
        """
        self.num_hashes = num_hashes
        self.hash_functions = self.generate_hash_functions()

    def generate_hash_functions(self):
        """
        Generate a list of hash functions using a hash family.
        """
        hash_functions = []
        for i in range(self.num_hashes):
            hash_functions.append(self.hash_family(i))
        return hash_functions

    def hash_family(self, i):
        """
        Generate a specific hash function based on an index.
        """
        result_size = 8 
        max_len = 20
        salt = str(i).zfill(max_len)[-max_len:]

        def hash_member(x):
            return int(hashlib.sha1((x + salt).encode('utf-8')).hexdigest()[-result_size:], 16)
        return hash_member

    def get_signature(self, shingles):
        """
        Compute the MinHash signature for a set of shingles.
        """
        signature = []
        for hf in self.hash_functions:
            min_hash = min(hf(shingle) for shingle in shingles)
            signature.append(min_hash)
        return signature


class LSH:
    """
    A class for Locality Sensitive Hashing for finding similar documents.
    """
    def __init__(self, signatures, r, b):
        """
        Initialize the LSH class with a list of MinHash signatures, 
        number of rows per band and Number of bands.
        """
        self.signatures = signatures
        self.r = r
        self.b = b
        self.buckets = self._initialize_buckets(b)

    def _initialize_buckets(self, b):
        """
        Create empty buckets, lists of empty dictionaries, for each band.
        """
        return [{} for _ in range(b)]

    def _hash_band(self, band):
        """
        Compute the hash of a given band of a signature.
        """
        return hashlib.sha1(''.join(map(str, band)).encode('utf-8')).hexdigest()

    def compute_lsh(self):
        """
        Find candidate document pairs by hashing bands into buckets and comparing signatures in the same bucket.
        """
        candidates = set()

        for doc_id, signature in enumerate(self.signatures):
            
            for band_idx in range(self.b):
                # Extract the current band
                band = signature[band_idx * self.r : (band_idx + 1) * self.r]
                # Hash the band 
                band_hash = self._hash_band(band)
                # Retrieve or create the bucket for the band hash
                bucket = self.buckets[band_idx].setdefault(band_hash, [])
                # Compare with documents in the same bucket and add candidate pairs
                candidates.update((min(doc_id, candidate_id), max(doc_id, candidate_id)) 
                                for candidate_id in bucket if candidate_id != doc_id)

                # Add the current document to the bucket
                bucket.append(doc_id)
        return candidates


class NearDuplicatesJS:
    """
    A class for finding near-duplicate documents using exact Jaccard Similarity.
    """
    def __init__(self, shingle_sets):
        """
        Initialize the class with the sets of shingles for each document.
        """
        self.shingle_sets = shingle_sets

    def compute_jaccard_similarity(self, set1, set2):
        """
        Calculate the Jaccard similarity between two sets.
        """
        intersection_size = len(set1.intersection(set2))
        union_size = len(set1.union(set2))
        return intersection_size / union_size if union_size > 0 else 0.0

    def find_nearest_neighbors(self, threshold):
        """
        Identify pairs of documents with a Jaccard similarity greater than or equal to the threshold.
        """
        nearest_neighbors = set()
        num_docs = len(self.shingle_sets)

        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                similarity = self.compute_jaccard_similarity(self.shingle_sets[i], self.shingle_sets[j])
                if similarity >= threshold:
                    nearest_neighbors.add((i, j))

        return nearest_neighbors


if __name__ == "__main__":
    # Parameters
    k = 10  
    num_hashes = 120  
    threshold = 0.8  
    r = 5  
    b = num_hashes // r  

    # Read and preprocess documents by looking at their lenght
    unique_documents = set()
    with open('products.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            description = line.split('\t')[0].strip()
            if len(description) >= k:
                unique_documents.add(description)

    documents = list(unique_documents)
    print(f"Unique documents with length > k=10: {len(documents)}")

    # Shingling
    shingler = Shinglings(k)
    shingle_sets = [shingler.create_shingles(doc) for doc in documents]

    # MinHashing
    min_hash = MinHashing(num_hashes)
    signatures = [min_hash.get_signature(shingles) for shingles in shingle_sets]

    # Locality Sensitive Hashing
    lsh = LSH(signatures, r, b)
    start_time = time.time()
    lsh_pairs = lsh.compute_lsh()
    lsh_time = time.time() - start_time

    print(f"Pairs found by LSH: {len(lsh_pairs)}")
    print(f"Time taken by LSH: {lsh_time:.2f} seconds")

    # Stampa tutte le coppie candidate trovate con LSH
    print("\nDocument pairs found by LSH\n")
    for i, j in lsh_pairs:
        print(f"Document {i}: {documents[i]}")
        print(f"Document {j}: {documents[j]}\n")

    # Computing brute-force Jaccard Similarity
    JS = NearDuplicatesJS(shingle_sets)
    start_time = time.time()
    js_nearest_neighbors = JS.find_nearest_neighbors(threshold)
    js_time = time.time() - start_time

    print(f"\nDocument found by Jaccard Similarity: {len(js_nearest_neighbors)}")
    print(f"Time taken by computing brute force Jaccard Similarity: {js_time:.2f} seconds\n")
    for d1, d2 in js_nearest_neighbors:
        print(f"Document 1: {documents[d1]}")
        print(f"Document 2: {documents[d2]}\n")

    # Verify LSH candidates with Jaccard Similarity
    lsh_near_duplicates = set()
    print("\nLSH Candidate Pairs by their Jaccard Similarity:")
    for i, j in lsh_pairs:
        similarity = JS.compute_jaccard_similarity(shingle_sets[i], shingle_sets[j])
        print(f"Candidate Pair: ({i}, {j}) | Jaccard Similarity: {similarity:.2f}")
        if similarity >= threshold:
            lsh_near_duplicates.add((i, j))

    # Intersection of results
    intersection = lsh_near_duplicates.intersection(js_nearest_neighbors)
    print(f"\nIntersection size: {len(intersection)}")
