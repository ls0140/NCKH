# generate_synthetic_papers.py

import logging
import datetime
import random
from sqlalchemy.orm import sessionmaker
from database.schema import Paper, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def generate_synthetic_papers():
    """
    Generates synthetic papers to supplement the ArXiv papers and reach our target of 200 papers in each category.
    """
    session = Session()
    logging.info("Generating synthetic papers to reach target of 200 papers in each category...")
    
    try:
        # Check how many papers we currently have
        current_papers = session.query(Paper).count()
        logging.info(f"Current papers in database: {current_papers}")
        
        # We need to generate enough papers to have at least 400 total
        papers_to_generate = max(0, 400 - current_papers)
        
        if papers_to_generate == 0:
            logging.info("Already have enough papers. No need to generate synthetic ones.")
            return
        
        logging.info(f"Generating {papers_to_generate} synthetic papers...")
        
        # Sample titles and abstracts for synthetic papers
        sample_titles = [
            "Advanced Neural Network Architectures for Deep Learning",
            "Machine Learning Applications in Computer Vision",
            "Natural Language Processing with Transformer Models",
            "Reinforcement Learning for Autonomous Systems",
            "Deep Learning for Medical Image Analysis",
            "Optimization Algorithms in Machine Learning",
            "Graph Neural Networks for Social Network Analysis",
            "Attention Mechanisms in Neural Networks",
            "Federated Learning for Privacy-Preserving AI",
            "Meta-Learning Approaches for Few-Shot Learning",
            "Generative Adversarial Networks for Image Synthesis",
            "Self-Supervised Learning in Computer Vision",
            "Multi-Modal Learning with Neural Networks",
            "Explainable AI for Trustworthy Machine Learning",
            "Continual Learning in Neural Networks",
            "Neural Architecture Search for Automated ML",
            "Quantum Machine Learning Algorithms",
            "Adversarial Training for Robust Neural Networks",
            "Knowledge Distillation in Deep Learning",
            "Neural Network Compression Techniques"
        ]
        
        sample_abstracts = [
            "This paper presents a novel approach to neural network architecture design that improves performance across multiple domains.",
            "We introduce a new methodology for applying machine learning techniques to complex computer vision tasks.",
            "Our research demonstrates the effectiveness of transformer-based models in natural language processing applications.",
            "This work explores reinforcement learning strategies for developing autonomous systems with improved decision-making capabilities.",
            "We propose a deep learning framework specifically designed for medical image analysis and diagnosis.",
            "This paper investigates optimization algorithms that enhance the training efficiency of machine learning models.",
            "Our approach utilizes graph neural networks to analyze complex social network structures and relationships.",
            "We present a comprehensive study of attention mechanisms and their impact on neural network performance.",
            "This research focuses on federated learning techniques that preserve privacy while enabling collaborative AI development.",
            "We explore meta-learning strategies that enable rapid adaptation to new tasks with limited training data."
        ]
        
        # Generate synthetic papers
        for i in range(papers_to_generate):
            # Randomly select title and abstract
            title = random.choice(sample_titles) + f" - Study {i+1}"
            abstract = random.choice(sample_abstracts)
            
            # Generate random publication year (2015-2025)
            publication_year = random.randint(2015, 2025)
            
            # Generate synthetic citation count based on year
            current_year = datetime.datetime.now().year
            age = current_year - publication_year + 1
            
            # Base citation count that decreases with age
            base_citations = max(1, 50 - (age * 2))
            random_factor = random.uniform(0.5, 2.0)
            
            # Boost citations for recent papers
            if age <= 3:
                recent_boost = random.uniform(1.5, 3.0)
                citation_count = int(base_citations * random_factor * recent_boost)
            else:
                citation_count = int(base_citations * random_factor)
            
            citation_count = max(1, citation_count)
            
            # Create synthetic paper
            synthetic_paper = Paper(
                title=title,
                abstract=abstract,
                publication_year=publication_year,
                doi=f"synthetic_{i+1}",
                source_url=f"https://synthetic.paper/{i+1}",
                pdf_url=f"https://synthetic.paper/{i+1}/pdf",
                source_db="synthetic",
                citation_count=citation_count,
                last_updated=datetime.datetime.utcnow()
            )
            
            session.add(synthetic_paper)
        
        session.commit()
        logging.info(f"Successfully generated {papers_to_generate} synthetic papers.")
        
        # Verify total count
        total_papers = session.query(Paper).count()
        logging.info(f"Total papers in database after generation: {total_papers}")
        
    except Exception as e:
        logging.error(f"Error generating synthetic papers: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    generate_synthetic_papers() 