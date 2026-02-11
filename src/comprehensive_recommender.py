"""
COMPREHENSIVE RECOMMENDER SYSTEM
Demonstrates: Content-Based, Collaborative Filtering, Network Link Analysis, 
Personalized Recommendations, and Hybrid Systems

This code shows WHERE and HOW each concept is used in a complete system.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import defaultdict

# ==============================================================================
# SAMPLE DATA
# ==============================================================================

# Users dataset
users_data = {
    'userid': [1, 2, 3, 4, 5, 6],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC'],
    'age': [25, 30, 25, 35, 28, 32]
}

# Books dataset
books_data = {
    'ISBN': ['B001', 'B002', 'B003', 'B004', 'B005', 'B006'],
    'name': ['Python Programming', 'Data Science Basics', 'Machine Learning', 
             'Deep Learning', 'Statistics 101', 'AI Fundamentals'],
    'year': [2020, 2019, 2021, 2022, 2018, 2021],
    'author': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Wilson', 'Jane Smith', 'Bob Wilson'],
    'publisher': ['TechBooks', 'DataPub', 'TechBooks', 'AIPub', 'DataPub', 'AIPub']
}

# User-Item interactions (ratings)
ratings_data = {
    'userid': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6],
    'ISBN': ['B001', 'B002', 'B003', 'B001', 'B004', 'B002', 'B003', 'B005', 
             'B004', 'B006', 'B001', 'B005', 'B003', 'B006'],
    'rating': [5, 4, 5, 5, 4, 4, 5, 3, 5, 4, 3, 4, 5, 5]
}

users_df = pd.DataFrame(users_data)
books_df = pd.DataFrame(books_data)
ratings_df = pd.DataFrame(ratings_data)

print("="*80)
print("COMPREHENSIVE RECOMMENDER SYSTEM - ALL CONCEPTS DEMONSTRATED")
print("="*80)


# ==============================================================================
# 1. CONTENT-BASED FILTERING
# ==============================================================================
print("\n" + "="*80)
print("1. CONTENT-BASED FILTERING")
print("="*80)
print("Uses: Book features (author, publisher, year)")
print("Where in code: Creating item profiles and comparing with user preferences")

class ContentBasedRecommender:
    def __init__(self, books_df, ratings_df):
        self.books_df = books_df
        self.ratings_df = ratings_df
        self.item_profiles = None
        
    def create_item_profiles(self):
        """Create feature vectors for each book"""
        # Combine text features
        self.books_df['content'] = (self.books_df['author'] + ' ' + 
                                    self.books_df['publisher'] + ' ' + 
                                    self.books_df['year'].astype(str))
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer()
        self.item_profiles = tfidf.fit_transform(self.books_df['content'])
        
        print("\n📚 Item Profiles Created (TF-IDF vectors)")
        print(f"Shape: {self.item_profiles.shape}")
        return self.item_profiles
    
    def get_recommendations(self, user_id, top_n=3):
        """Get content-based recommendations for a user"""
        # Get books user has rated highly (rating >= 4)
        user_ratings = self.ratings_df[self.ratings_df['userid'] == user_id]
        liked_books = user_ratings[user_ratings['rating'] >= 4]['ISBN'].values
        
        if len(liked_books) == 0:
            return []
        
        # Get indices of liked books
        liked_indices = [self.books_df[self.books_df['ISBN'] == isbn].index[0] 
                        for isbn in liked_books]
        
        # Create user profile (average of liked items)
        user_profile = np.asarray(self.item_profiles[liked_indices].mean(axis=0))
        
        # Calculate similarity with all items
        similarities = cosine_similarity(user_profile, self.item_profiles)[0]
        
        # Get books user hasn't rated
        rated_isbns = user_ratings['ISBN'].values
        unrated_mask = ~self.books_df['ISBN'].isin(rated_isbns)
        unrated_indices = self.books_df[unrated_mask].index
        
        # Get top recommendations
        unrated_scores = [(idx, similarities[idx]) for idx in unrated_indices]
        unrated_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, score in unrated_scores[:top_n]:
            recommendations.append({
                'ISBN': self.books_df.iloc[idx]['ISBN'],
                'name': self.books_df.iloc[idx]['name'],
                'score': score
            })
        
        return recommendations

cb_recommender = ContentBasedRecommender(books_df, ratings_df)
cb_recommender.create_item_profiles()

print("\n🎯 Content-Based Recommendations for User 1:")
cb_recs = cb_recommender.get_recommendations(1, top_n=2)
for rec in cb_recs:
    print(f"  - {rec['name']} (Score: {rec['score']:.3f})")


# ==============================================================================
# 2. COLLABORATIVE FILTERING
# ==============================================================================
print("\n" + "="*80)
print("2. COLLABORATIVE FILTERING")
print("="*80)
print("Uses: User-item rating patterns (finds similar users/items)")
print("Where in code: User-user or item-item similarity matrices")

class CollaborativeFilteringRecommender:
    def __init__(self, ratings_df, users_df, books_df):
        self.ratings_df = ratings_df
        self.users_df = users_df
        self.books_df = books_df
        self.user_item_matrix = None
        
    def create_user_item_matrix(self):
        """Create user-item rating matrix"""
        self.user_item_matrix = self.ratings_df.pivot(
            index='userid', 
            columns='ISBN', 
            values='rating'
        ).fillna(0)
        
        print("\n📊 User-Item Matrix:")
        print(self.user_item_matrix)
        return self.user_item_matrix
    
    def user_based_cf(self, user_id, top_n=3):
        """User-based collaborative filtering"""
        # Calculate user-user similarity
        user_similarity = cosine_similarity(self.user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print(f"\n👥 User Similarity Matrix (for user {user_id}):")
        print(user_similarity_df.loc[user_id].sort_values(ascending=False).head(3))
        
        # Get similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:4]
        
        # Get recommendations from similar users
        recommendations = {}
        for sim_user, similarity in similar_users.items():
            sim_user_ratings = self.user_item_matrix.loc[sim_user]
            for isbn, rating in sim_user_ratings.items():
                if rating > 0 and self.user_item_matrix.loc[user_id, isbn] == 0:
                    if isbn not in recommendations:
                        recommendations[isbn] = 0
                    recommendations[isbn] += similarity * rating
        
        # Sort and get top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        result = []
        for isbn, score in sorted_recs:
            book_name = self.books_df[self.books_df['ISBN'] == isbn]['name'].values[0]
            result.append({'ISBN': isbn, 'name': book_name, 'score': score})
        
        return result

cf_recommender = CollaborativeFilteringRecommender(ratings_df, users_df, books_df)
cf_recommender.create_user_item_matrix()

print("\n🎯 Collaborative Filtering Recommendations for User 1:")
cf_recs = cf_recommender.user_based_cf(1, top_n=2)
for rec in cf_recs:
    print(f"  - {rec['name']} (Score: {rec['score']:.3f})")


# ==============================================================================
# 3. NETWORK LINK ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("3. NETWORK LINK ANALYSIS (GRAPH-BASED)")
print("="*80)
print("Uses: User-item interactions as a graph/network")
print("Where in code: Build graph, calculate centrality, find important items")
print("Connection to RecSys: Identifies influential items and user communities")

class NetworkLinkAnalysis:
    def __init__(self, ratings_df, books_df, users_df):
        self.ratings_df = ratings_df
        self.books_df = books_df
        self.users_df = users_df
        self.graph = nx.Graph()
        
    def build_bipartite_graph(self):
        """Build user-item bipartite graph"""
        # Add nodes
        for user_id in self.users_df['userid']:
            self.graph.add_node(f"U{user_id}", node_type='user')
        
        for isbn in self.books_df['ISBN']:
            self.graph.add_node(f"I{isbn}", node_type='item')
        
        # Add edges (with ratings as weights)
        for _, row in self.ratings_df.iterrows():
            self.graph.add_edge(
                f"U{row['userid']}", 
                f"I{row['ISBN']}", 
                weight=row['rating']
            )
        
        print(f"\n🕸️  Bipartite Graph Built:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def calculate_pagerank(self):
        """Calculate PageRank to find important items"""
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Extract item scores
        item_scores = {k.replace('I', ''): v for k, v in pagerank.items() 
                       if k.startswith('I')}
        
        print("\n📈 PageRank Scores (Item Importance):")
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        for isbn, score in sorted_items[:3]:
            book_name = self.books_df[self.books_df['ISBN'] == isbn]['name'].values[0]
            print(f"   {book_name}: {score:.4f}")
        
        return item_scores
    
    def get_recommendations_via_random_walk(self, user_id, top_n=3):
        """Recommend items via personalized random walk from user node"""
        # Personalized PageRank starting from user
        personalization = {f"U{user_id}": 1.0}
        ppr = nx.pagerank(self.graph, personalization=personalization, weight='weight')
        
        # Get item scores
        item_scores = {k.replace('I', ''): v for k, v in ppr.items() 
                       if k.startswith('I')}
        
        # Remove already rated items
        rated_items = self.ratings_df[self.ratings_df['userid'] == user_id]['ISBN'].values
        item_scores = {k: v for k, v in item_scores.items() if k not in rated_items}
        
        # Get top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommendations = []
        for isbn, score in sorted_items:
            book_name = self.books_df[self.books_df['ISBN'] == isbn]['name'].values[0]
            recommendations.append({'ISBN': isbn, 'name': book_name, 'score': score})
        
        return recommendations

network_analyzer = NetworkLinkAnalysis(ratings_df, books_df, users_df)
network_analyzer.build_bipartite_graph()
network_analyzer.calculate_pagerank()

print(f"\n🎯 Network-Based Recommendations for User 1 (via Random Walk):")
network_recs = network_analyzer.get_recommendations_via_random_walk(1, top_n=2)
for rec in network_recs:
    print(f"  - {rec['name']} (Score: {rec['score']:.4f})")


# ==============================================================================
# 4. PERSONALIZED RECOMMENDATIONS
# ==============================================================================
print("\n" + "="*80)
print("4. PERSONALIZED RECOMMENDATIONS")
print("="*80)
print("Uses: User demographics (age, city) + preferences")
print("Where in code: Incorporate user features to customize recommendations")
print("Note: This ENHANCES other methods with user-specific context")

class PersonalizedRecommender:
    def __init__(self, users_df, books_df, ratings_df):
        self.users_df = users_df
        self.books_df = books_df
        self.ratings_df = ratings_df
    
    def get_user_context(self, user_id):
        """Extract user demographics and preferences"""
        user_info = self.users_df[self.users_df['userid'] == user_id].iloc[0]
        user_ratings = self.ratings_df[self.ratings_df['userid'] == user_id]
        
        context = {
            'age': user_info['age'],
            'city': user_info['city'],
            'avg_rating': user_ratings['rating'].mean(),
            'num_ratings': len(user_ratings)
        }
        
        print(f"\n👤 User {user_id} Context (for personalization):")
        print(f"   Age: {context['age']}, City: {context['city']}")
        print(f"   Avg Rating: {context['avg_rating']:.2f}, # Ratings: {context['num_ratings']}")
        
        return context
    
    def personalize_by_demographics(self, user_id, base_recommendations, top_n=3):
        """Apply demographic filters to personalize recommendations"""
        user_context = self.get_user_context(user_id)
        
        # Example: Boost recent books for younger users
        personalized_recs = []
        for rec in base_recommendations:
            isbn = rec['ISBN']
            book_year = self.books_df[self.books_df['ISBN'] == isbn]['year'].values[0]
            
            # Personalization logic
            boost = 1.0
            if user_context['age'] < 30 and book_year >= 2020:
                boost = 1.2  # Boost recent books for young users
            
            personalized_score = rec['score'] * boost
            personalized_recs.append({
                'ISBN': isbn,
                'name': rec['name'],
                'original_score': rec['score'],
                'personalized_score': personalized_score,
                'boost_applied': boost
            })
        
        # Re-sort by personalized score
        personalized_recs.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        return personalized_recs[:top_n]

personalizer = PersonalizedRecommender(users_df, books_df, ratings_df)

print("\n🎯 Personalized Recommendations for User 1 (based on CF):")
personalized_recs = personalizer.personalize_by_demographics(1, cf_recs, top_n=2)
for rec in personalized_recs:
    print(f"  - {rec['name']}")
    print(f"    Original: {rec['original_score']:.3f}, " +
          f"Personalized: {rec['personalized_score']:.3f}, " +
          f"Boost: {rec['boost_applied']:.1f}x")


# ==============================================================================
# 5. HYBRID RECOMMENDATION SYSTEM
# ==============================================================================
print("\n" + "="*80)
print("5. HYBRID RECOMMENDATION SYSTEM")
print("="*80)
print("Uses: Combines ALL previous methods")
print("Where in code: Weighted combination of different recommendation scores")

class HybridRecommender:
    def __init__(self, cb_recommender, cf_recommender, network_analyzer, personalizer):
        self.cb_recommender = cb_recommender
        self.cf_recommender = cf_recommender
        self.network_analyzer = network_analyzer
        self.personalizer = personalizer
    
    def get_hybrid_recommendations(self, user_id, top_n=3, 
                                   w_content=0.3, w_collab=0.3, w_network=0.2, w_personal=0.2):
        """
        Combine all methods with weights
        
        Parameters:
        - w_content: Weight for content-based
        - w_collab: Weight for collaborative filtering
        - w_network: Weight for network analysis
        - w_personal: Weight for personalization boost
        """
        
        print(f"\n🔀 Hybrid Combination Weights:")
        print(f"   Content-Based: {w_content}")
        print(f"   Collaborative: {w_collab}")
        print(f"   Network-Based: {w_network}")
        print(f"   Personalization: {w_personal}")
        
        # Get recommendations from each method
        cb_recs = self.cb_recommender.get_recommendations(user_id, top_n=10)
        cf_recs = self.cf_recommender.user_based_cf(user_id, top_n=10)
        net_recs = self.network_analyzer.get_recommendations_via_random_walk(user_id, top_n=10)
        
        # Normalize scores and combine
        hybrid_scores = defaultdict(lambda: {'score': 0, 'name': '', 'components': {}})
        
        # Add content-based scores
        if cb_recs:
            max_cb = max([r['score'] for r in cb_recs])
            for rec in cb_recs:
                isbn = rec['ISBN']
                normalized_score = (rec['score'] / max_cb) * w_content
                hybrid_scores[isbn]['score'] += normalized_score
                hybrid_scores[isbn]['name'] = rec['name']
                hybrid_scores[isbn]['components']['content'] = normalized_score
        
        # Add collaborative scores
        if cf_recs:
            max_cf = max([r['score'] for r in cf_recs])
            for rec in cf_recs:
                isbn = rec['ISBN']
                normalized_score = (rec['score'] / max_cf) * w_collab
                hybrid_scores[isbn]['score'] += normalized_score
                hybrid_scores[isbn]['name'] = rec['name']
                hybrid_scores[isbn]['components']['collab'] = normalized_score
        
        # Add network scores
        if net_recs:
            max_net = max([r['score'] for r in net_recs])
            for rec in net_recs:
                isbn = rec['ISBN']
                normalized_score = (rec['score'] / max_net) * w_network
                hybrid_scores[isbn]['score'] += normalized_score
                hybrid_scores[isbn]['name'] = rec['name']
                hybrid_scores[isbn]['components']['network'] = normalized_score
        
        # Apply personalization boost
        user_context = self.personalizer.get_user_context(user_id)
        for isbn in hybrid_scores:
            book_year = self.cb_recommender.books_df[
                self.cb_recommender.books_df['ISBN'] == isbn
            ]['year'].values[0]
            
            boost = 1.0
            if user_context['age'] < 30 and book_year >= 2020:
                boost = 1.0 + w_personal
            
            hybrid_scores[isbn]['score'] *= boost
            hybrid_scores[isbn]['components']['personal_boost'] = boost
        
        # Sort and return top N
        sorted_recs = sorted(hybrid_scores.items(), 
                           key=lambda x: x[1]['score'], 
                           reverse=True)[:top_n]
        
        recommendations = []
        for isbn, data in sorted_recs:
            recommendations.append({
                'ISBN': isbn,
                'name': data['name'],
                'hybrid_score': data['score'],
                'components': data['components']
            })
        
        return recommendations

# Create hybrid recommender
hybrid_recommender = HybridRecommender(
    cb_recommender, 
    cf_recommender, 
    network_analyzer, 
    personalizer
)

print("\n🎯 FINAL HYBRID RECOMMENDATIONS for User 1:")
hybrid_recs = hybrid_recommender.get_hybrid_recommendations(1, top_n=3)

for i, rec in enumerate(hybrid_recs, 1):
    print(f"\n{i}. {rec['name']} (Hybrid Score: {rec['hybrid_score']:.4f})")
    print(f"   Score breakdown:")
    for component, value in rec['components'].items():
        print(f"     - {component}: {value:.4f}")


# ==============================================================================
# SUMMARY: WHERE EACH CONCEPT IS USED
# ==============================================================================
print("\n" + "="*80)
print("📋 SUMMARY: WHERE TO POINT IN YOUR CODE")
print("="*80)

summary = """
1. CONTENT-BASED FILTERING:
   ✓ Class: ContentBasedRecommender
   ✓ Key method: create_item_profiles() - Creates TF-IDF vectors from item features
   ✓ Show: Item profile matrix and cosine similarity calculation
   
2. COLLABORATIVE FILTERING:
   ✓ Class: CollaborativeFilteringRecommender
   ✓ Key method: create_user_item_matrix() - Creates rating matrix
   ✓ Key method: user_based_cf() - Calculates user similarities
   ✓ Show: User-user similarity matrix
   
3. NETWORK LINK ANALYSIS:
   ✓ Class: NetworkLinkAnalysis
   ✓ Key method: build_bipartite_graph() - Creates user-item graph
   ✓ Key method: calculate_pagerank() - Finds influential items
   ✓ Key method: get_recommendations_via_random_walk() - Graph-based recommendations
   ✓ Show: NetworkX graph object and PageRank scores
   ✓ EXPLAIN: "This finds important books and user communities through graph structure"
   
4. PERSONALIZED RECOMMENDATIONS:
   ✓ Class: PersonalizedRecommender
   ✓ Key method: get_user_context() - Extracts user demographics
   ✓ Key method: personalize_by_demographics() - Applies user-specific boosts
   ✓ Show: User context dictionary and boost factors
   ✓ EXPLAIN: "This is NOT a separate algorithm - it ENHANCES other methods with user info"
   
5. HYBRID SYSTEM:
   ✓ Class: HybridRecommender
   ✓ Key method: get_hybrid_recommendations() - Combines all methods
   ✓ Show: Weighted combination formula and component scores
   ✓ EXPLAIN: "This combines content-based + collaborative + network + personalization"
"""

print(summary)

print("\n" + "="*80)
print("🎓 HOW TO ANSWER YOUR LECTURER")
print("="*80)

answer_guide = """
Q: "Where is network link analysis in your code?"
A: Point to NetworkLinkAnalysis class. Explain:
   - Built bipartite graph from user-item interactions
   - Used PageRank to identify important books
   - Used personalized random walk for recommendations
   - This captures network effects (popular items, community influence)

Q: "Where is personalization?"
A: Point to PersonalizedRecommender class. Explain:
   - This is not a standalone algorithm
   - It ENHANCES recommendations using user demographics (age, city)
   - Applied as boost factors to other methods' scores
   - See personalize_by_demographics() and boost calculation in hybrid system

Q: "Show me the hybrid part"
A: Point to HybridRecommender.get_hybrid_recommendations(). Explain:
   - Combines content-based, collaborative, network, and personalization
   - Uses weighted averaging (you can adjust weights)
   - Each component contributes to final score
   - Show the 'components' breakdown in output
"""

print(answer_guide)

print("\n" + "="*80)
print("✅ CODE EXECUTION COMPLETE")
print("="*80)
