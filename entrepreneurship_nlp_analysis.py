import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import networkx as nx
from collections import Counter
import re
from scipy.stats import entropy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EntrepreneurshipNLPAnalyzer:
    def __init__(self, data_path):
        """Initialize the NLP analyzer with competency definitions and data loading"""
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
        
        # Initialize text preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Define the 10 competencies and their keywords
        self.competencies = {
            'Innovation': ['creativity', 'ideation', 'design thinking', 'problem solving', 
                          'creative solutions', 'innovative approaches', 'breakthrough thinking', 
                          'disruptive innovation'],
            
            'Entrepreneurship': ['venture creation', 'business model', 'startup', 'business plan', 
                                'market research', 'feasibility study', 'business strategy', 
                                'value proposition', 'revenue model', 'business development'],
            
            'Social Impact': ['social enterprise', 'sustainability', 'community development', 
                             'social innovation', 'triple bottom line', 'social responsibility', 
                             'impact measurement', 'social venture', 'community empowerment', 
                             'environmental impact'],
            
            'Leadership': ['team building', 'collaboration', 'project management', 'communication', 
                          'delegation', 'motivation', 'team dynamics', 'interpersonal skills', 
                          'conflict resolution', 'strategic thinking'],
            
            'Financial Literacy': ['budgeting', 'financial planning', 'investment', 'funding', 
                                  'venture capital', 'angel investors', 'financial statements', 
                                  'cash flow', 'cost analysis', 'fundraising'],
            
            'Marketing': ['marketing strategy', 'customer acquisition', 'branding', 'digital marketing', 
                         'sales techniques', 'customer relationship management', 'market segmentation', 
                         'promotional strategies', 'customer validation', 'market analysis'],
            
            'Risk Management': ['risk assessment', 'uncertainty', 'decision making', 'contingency planning', 
                               'risk mitigation', 'strategic planning', 'adaptability', 'resilience', 
                               'crisis management', 'uncertainty management'],
            
            'Technology': ['digital transformation', 'technology adoption', 'e-commerce', 'digital platforms', 
                          'automation', 'data analytics', 'digital tools', 'online presence', 
                          'technological innovation', 'digital literacy'],
            
            'Networking': ['networking', 'partnerships', 'stakeholder engagement', 'relationship building', 
                          'mentorship', 'advisory boards', 'strategic alliances', 'community engagement', 
                          'professional relationships', 'collaboration networks'],
            
            'Ethics': ['business ethics', 'corporate responsibility', 'ethical decision making', 
                      'integrity', 'transparency', 'accountability', 'stakeholder responsibility', 
                      'ethical leadership', 'moral reasoning', 'social responsibility']
        }
        
        # Initialize BART-MNLI classifier
        print("Loading BART-MNLI model...")
        self.classifier = pipeline("zero-shot-classification", 
                                  model="facebook/bart-large-mnli",
                                  device=-1)  # Use CPU
        
        # Load and preprocess data
        self.load_data(data_path)
        
        # Analysis columns
        self.text_columns = [
            'Undergrad Entrepreneurship Courses & Programs ',
            'Teaching and Assessment Methods',
            'Co-Curricular Activities',
            'Annual Report + Enactus Impact Data'
        ]
        
        # Results storage
        self.competency_scores = {}
        self.analysis_results = {}
        
    def load_data(self, data_path):
        """Load and clean the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(data_path)
        
        # Remove empty rows and basic cleaning
        self.df = self.df.dropna(subset=['School Name'])
        self.df = self.df[self.df['School Name'].str.strip() != '']
        
        print(f"Loaded {len(self.df)} institutions")
        print(f"Columns: {list(self.df.columns)}")
        
    def preprocess_text(self, text):
        """Clean and preprocess text by removing stopwords, punctuation, and lemmatizing"""
        if pd.isna(text) or text.strip() == '':
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords, punctuation, and short words
        filtered_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                token not in string.punctuation and 
                len(token) > 2 and
                token.isalpha()):
                # Lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)
        
        # Join back into text
        cleaned_text = ' '.join(filtered_tokens)
        return cleaned_text
        
    def classify_competency_bart(self, text, competency_name, keywords):
        """Use BART-MNLI to classify text for a specific competency using hypothesis-based NLI"""
        if pd.isna(text) or text.strip() == '':
            return 0.0
            
        # Preprocess text to remove stopwords and clean it
        cleaned_text = self.preprocess_text(text)
        
        # If cleaning resulted in empty text, return 0
        if not cleaned_text.strip():
            return 0.0
        
        try:
            # Method 1: Direct competency classification
            competency_labels = [competency_name]
            result1 = self.classifier(cleaned_text, competency_labels)
            main_score = result1['scores'][0] if result1['scores'] else 0.0

            # # Method 2: Keyword-based classification (more granular)
            # keyword_labels = keywords[:6]  # Use top 6 keywords
            # if keyword_labels:
            #     result2 = self.classifier(cleaned_text, keyword_labels)
            #     # Average the top 3 keyword scores
            #     top_keyword_scores = sorted(result2['scores'], reverse=True)[:3]
            #     keyword_score = np.mean(top_keyword_scores) if top_keyword_scores else 0.0
            # else:
            #     keyword_score = 0.0

            # # Method 3: Hypothesis-based approach (most explicit)
            # hypothesis = f"This text discusses {competency_name.lower()} concepts and methods"
            # try:
            #     # Use entailment classification directly
            #     entailment_result = self.classifier(cleaned_text, [f"{competency_name} concepts", "unrelated content"])
            #     hypothesis_score = entailment_result['scores'][0] if entailment_result['scores'] else 0.0
            # except:
            #     hypothesis_score = 0.0

            # Use only direct competency classification
            final_score = main_score

            # # Combine scores with weights (COMMENTED OUT - now using only direct classification)
            # # Main competency: 50%, Keywords: 30%, Hypothesis: 20%
            # final_score = (0.5 * main_score + 0.3 * keyword_score + 0.2 * hypothesis_score)

            # Debug info (can be removed later)
            if final_score > 0.1:  # Only print for significant scores
                print(f"  {competency_name}: score={final_score:.3f}")
                # print(f"  {competency_name}: main={main_score:.3f}, keywords={keyword_score:.3f}, hypothesis={hypothesis_score:.3f}, final={final_score:.3f}")
                
            return final_score
            
        except Exception as e:
            print(f"Error classifying text for {competency_name}: {str(e)[:100]}")
            return 0.0
    
    def calculate_competency_scores(self):
        """Calculate competency scores for all institutions across all text fields"""
        print("Calculating competency scores using BART-MNLI...")
        
        results = []
        
        for idx, row in self.df.iterrows():
            institution = row['School Name']
            region = row.get('Enactus Region', 'Unknown')
            
            print(f"Processing {institution}...")
            
            institution_scores = {
                'Institution': institution,
                'Region': region
            }
            
            # Process each text column
            for col in self.text_columns:
                text = str(row.get(col, ''))
                
                # Calculate scores for each competency
                for comp_name, keywords in self.competencies.items():
                    score = self.classify_competency_bart(text, comp_name, keywords)
                    institution_scores[f"{comp_name}_{col}"] = score
            
            # Calculate aggregate scores per competency
            for comp_name in self.competencies.keys():
                col_scores = [institution_scores[f"{comp_name}_{col}"] for col in self.text_columns]
                institution_scores[f"{comp_name}_Overall"] = np.mean(col_scores)
            
            results.append(institution_scores)
        
        self.competency_scores_df = pd.DataFrame(results)
        return self.competency_scores_df
        
    def calculate_metrics(self):
        """Calculate advanced metrics for each institution"""
        print("Calculating advanced metrics...")
        
        metrics = []
        
        for idx, row in self.df.iterrows():
            institution = row['School Name']
            
            # Get competency scores for this institution
            inst_scores = self.competency_scores_df[
                self.competency_scores_df['Institution'] == institution
            ].iloc[0]
            
            # Extract overall competency scores
            overall_scores = [inst_scores[f"{comp}_Overall"] 
                            for comp in self.competencies.keys()]
            
            # Calculate diversity metrics
            diversity_index = entropy(np.array(overall_scores) + 1e-10)  # Shannon entropy
            comprehensiveness_score = np.mean(overall_scores)
            
            # Innovation Quotient (Innovation + Technology + Entrepreneurship)
            innovation_quotient = np.mean([
                inst_scores['Innovation_Overall'],
                inst_scores['Technology_Overall'], 
                inst_scores['Entrepreneurship_Overall']
            ])
            
            # Text complexity analysis
            all_text = ""
            for col in self.text_columns:
                text = str(row.get(col, ''))
                if text != 'nan':
                    all_text += " " + text
            
            readability_score = flesch_reading_ease(all_text) if all_text.strip() else 0
            grade_level = flesch_kincaid_grade(all_text) if all_text.strip() else 0
            text_length = len(all_text.split())
            
            metrics.append({
                'Institution': institution,
                'Diversity_Index': diversity_index,
                'Comprehensiveness_Score': comprehensiveness_score,
                'Innovation_Quotient': innovation_quotient,
                'Readability_Score': readability_score,
                'Grade_Level': grade_level,
                'Text_Length': text_length
            })
        
        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df
    
    def create_competency_heatmap(self):
        """Create heatmap showing competency distribution across institutions"""
        print("Creating competency distribution heatmap...")
        
        # Prepare data for heatmap
        heatmap_data = self.competency_scores_df.set_index('Institution')[
            [f"{comp}_Overall" for comp in self.competencies.keys()]
        ]
        
        # Rename columns to remove '_Overall'
        heatmap_data.columns = [col.replace('_Overall', '') for col in heatmap_data.columns]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f',
                   cbar_kws={'label': 'Competency Score'})
        plt.title('Competency Distribution Across Institutions\n(BART-MNLI Confidence Scores)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Competencies', fontsize=12)
        plt.ylabel('Institutions', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('viz/competency_distribution_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_program_area_heatmap(self):
        """Create heatmap showing where each competency is emphasized across program areas"""
        print("Creating program area emphasis heatmap...")
        
        # Calculate average scores for each competency across program areas
        program_area_data = []
        
        for comp in self.competencies.keys():
            row_data = {'Competency': comp}
            for col in self.text_columns:
                col_clean = col.strip()
                scores = self.competency_scores_df[f"{comp}_{col}"]
                row_data[col_clean] = scores.mean()
            program_area_data.append(row_data)
        
        program_df = pd.DataFrame(program_area_data).set_index('Competency')
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(program_df, annot=True, cmap='Blues', fmt='.3f',
                   cbar_kws={'label': 'Average Competency Score'})
        plt.title('Competency Emphasis Across Program Areas\n(Where Each Skill is Most Developed)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Program Areas', fontsize=12)
        plt.ylabel('Competencies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('viz/program_area_emphasis_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_regional_comparison_heatmap(self):
        """Create heatmap comparing competency focus across regions"""
        print("Creating regional comparison heatmap...")
        
        # Group by region and calculate mean competency scores
        regional_data = self.competency_scores_df.groupby('Region')[
            [f"{comp}_Overall" for comp in self.competencies.keys()]
        ].mean()
        
        # Rename columns
        regional_data.columns = [col.replace('_Overall', '') for col in regional_data.columns]
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(regional_data, annot=True, cmap='Greens', fmt='.3f',
                   cbar_kws={'label': 'Average Competency Score'})
        plt.title('Regional Competency Focus Comparison\n(Average Scores by Region)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Competencies', fontsize=12)
        plt.ylabel('Regions', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('viz/regional_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_performance_correlation(self):
        """Analyze correlation between competencies and Enactus performance"""
        print("Analyzing performance correlations...")
        
        # Extract performance indicators from the Annual Report column
        performance_data = []
        
        for idx, row in self.df.iterrows():
            institution = row['School Name']
            performance_text = str(row.get('Annual Report + Enactus Impact Data', ''))
            
            # Simple performance scoring based on keywords
            performance_score = 0
            
            # National level achievements
            if 'National Champion' in performance_text:
                performance_score += 10
            elif 'National' in performance_text and ('Champion' in performance_text or '1st place' in performance_text):
                performance_score += 8
            elif 'National' in performance_text:
                performance_score += 5
                
            # Regional achievements
            if 'Regional Champion' in performance_text:
                performance_score += 5
            elif 'Regional' in performance_text:
                performance_score += 2
                
            # Count number of achievements mentioned
            achievement_count = len(re.findall(r'Champion|1st place|2nd place|3rd place', performance_text))
            performance_score += achievement_count
            
            performance_data.append({
                'Institution': institution,
                'Performance_Score': performance_score
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Merge with competency scores
        merged_df = self.competency_scores_df.merge(performance_df, on='Institution')
        
        # Calculate correlations
        competency_cols = [f"{comp}_Overall" for comp in self.competencies.keys()]
        correlation_data = merged_df[competency_cols + ['Performance_Score']].corr()['Performance_Score'].drop('Performance_Score')
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = correlation_data.values.reshape(-1, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                   yticklabels=[col.replace('_Overall', '') for col in competency_cols],
                   xticklabels=['Performance Correlation'])
        plt.title('Competency-Performance Correlation Analysis\n(Correlation with Enactus Competition Success)', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('viz/performance_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_data
        
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with key insights"""
        print("Creating comprehensive insights dashboard...")
        
        # Create a large dashboard figure
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Top performers by different metrics
        ax1 = plt.subplot(4, 2, 1)
        top_comprehensive = self.metrics_df.nlargest(5, 'Comprehensiveness_Score')
        sns.barplot(data=top_comprehensive, y='Institution', x='Comprehensiveness_Score', ax=ax1)
        ax1.set_title('Top 5 Most Comprehensive Programs', fontweight='bold')
        ax1.set_xlabel('Comprehensiveness Score')
        
        # 2. Innovation leaders
        ax2 = plt.subplot(4, 2, 2)
        innovation_scores = self.competency_scores_df[['Institution', 'Innovation_Overall']].nlargest(5, 'Innovation_Overall')
        sns.barplot(data=innovation_scores, y='Institution', x='Innovation_Overall', ax=ax2, color='orange')
        ax2.set_title('Top 5 Innovation-Focused Programs', fontweight='bold')
        ax2.set_xlabel('Innovation Score')
        
        # 3. Competency diversity
        ax3 = plt.subplot(4, 2, 3)
        top_diverse = self.metrics_df.nlargest(5, 'Diversity_Index')
        sns.barplot(data=top_diverse, y='Institution', x='Diversity_Index', ax=ax3, color='green')
        ax3.set_title('Most Balanced Competency Portfolios', fontweight='bold')
        ax3.set_xlabel('Diversity Index (Shannon Entropy)')
        
        # 4. Social impact focus
        ax4 = plt.subplot(4, 2, 4)
        social_scores = self.competency_scores_df[['Institution', 'Social Impact_Overall']].nlargest(5, 'Social Impact_Overall')
        sns.barplot(data=social_scores, y='Institution', x='Social Impact_Overall', ax=ax4, color='purple')
        ax4.set_title('Top 5 Social Impact Programs', fontweight='bold')
        ax4.set_xlabel('Social Impact Score')
        
        # 5. Regional competency comparison (radar chart style)
        ax5 = plt.subplot(4, 2, 5)
        regional_means = self.competency_scores_df.groupby('Region')[
            [f"{comp}_Overall" for comp in self.competencies.keys()]
        ].mean()
        
        x_pos = np.arange(len(self.competencies))
        width = 0.25
        
        for i, region in enumerate(regional_means.index):
            ax5.bar(x_pos + i*width, regional_means.iloc[i].values, width, 
                   label=region, alpha=0.8)
        
        ax5.set_xlabel('Competencies')
        ax5.set_ylabel('Average Score')
        ax5.set_title('Regional Competency Profiles', fontweight='bold')
        ax5.set_xticks(x_pos + width)
        ax5.set_xticklabels([comp[:8] for comp in self.competencies.keys()], rotation=45)
        ax5.legend()
        
        # 6. Text complexity vs comprehensiveness
        ax6 = plt.subplot(4, 2, 6)
        scatter_data = self.metrics_df.merge(self.competency_scores_df[['Institution', 'Region']], on='Institution')
        sns.scatterplot(data=scatter_data, x='Text_Length', y='Comprehensiveness_Score', 
                       hue='Region', s=100, ax=ax6)
        ax6.set_title('Program Depth vs Text Complexity', fontweight='bold')
        ax6.set_xlabel('Text Length (words)')
        ax6.set_ylabel('Comprehensiveness Score')
        
        # 7. Competency correlation matrix
        ax7 = plt.subplot(4, 1, 4)
        comp_corr = self.competency_scores_df[[f"{comp}_Overall" for comp in self.competencies.keys()]].corr()
        mask = np.triu(np.ones_like(comp_corr, dtype=bool))
        sns.heatmap(comp_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=ax7,
                   xticklabels=[comp[:10] for comp in self.competencies.keys()],
                   yticklabels=[comp[:10] for comp in self.competencies.keys()])
        ax7.set_title('Competency Correlation Matrix\n(How competencies co-occur in programs)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('viz/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("Generating summary report...")
        
        report = []
        report.append("="*80)
        report.append("ENTREPRENEURSHIP EDUCATION NLP ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Top performers
        report.append("TOP PERFORMING INSTITUTIONS:")
        report.append("-" * 40)
        
        # Most comprehensive
        top_comp = self.metrics_df.nlargest(3, 'Comprehensiveness_Score')
        report.append("Most Comprehensive Programs:")
        for idx, row in top_comp.iterrows():
            report.append(f"  {row['Institution']}: {row['Comprehensiveness_Score']:.3f}")
        report.append("")
        
        # Highest innovation
        innovation_scores = self.competency_scores_df.nlargest(3, 'Innovation_Overall')
        report.append("Highest Innovation Focus:")
        for idx, row in innovation_scores.iterrows():
            report.append(f"  {row['Institution']}: {row['Innovation_Overall']:.3f}")
        report.append("")
        
        # Regional insights
        report.append("REGIONAL INSIGHTS:")
        report.append("-" * 40)
        regional_means = self.competency_scores_df.groupby('Region')[
            [f"{comp}_Overall" for comp in self.competencies.keys()]
        ].mean()
        
        for region in regional_means.index:
            report.append(f"{region} Region Strengths:")
            region_scores = regional_means.loc[region]
            top_competencies = region_scores.nlargest(3)
            for comp, score in top_competencies.items():
                clean_comp = comp.replace('_Overall', '')
                report.append(f"  {clean_comp}: {score:.3f}")
            report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        report.append("-" * 40)
        
        # Find most balanced institution
        most_balanced = self.metrics_df.loc[self.metrics_df['Diversity_Index'].idxmax()]
        report.append(f"Most Balanced Program: {most_balanced['Institution']} (Diversity Index: {most_balanced['Diversity_Index']:.3f})")
        
        # Find highest innovation quotient
        highest_iq = self.metrics_df.loc[self.metrics_df['Innovation_Quotient'].idxmax()]
        report.append(f"Highest Innovation Quotient: {highest_iq['Institution']} ({highest_iq['Innovation_Quotient']:.3f})")
        
        # Most readable content
        most_readable = self.metrics_df.loc[self.metrics_df['Readability_Score'].idxmax()]
        report.append(f"Most Accessible Content: {most_readable['Institution']} (Readability: {most_readable['Readability_Score']:.1f})")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        with open('viz/entrepreneurship_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        for line in report:
            print(line)
            
        return '\n'.join(report)
    
    def run_complete_analysis(self):
        """Run the complete NLP analysis pipeline"""
        print("Starting comprehensive NLP analysis...")
        print("="*60)
        
        # Step 1: Calculate competency scores
        self.calculate_competency_scores()
        
        # Step 2: Calculate advanced metrics
        self.calculate_metrics()
        
        # Step 3: Create visualizations
        self.create_competency_heatmap()
        self.create_program_area_heatmap()
        self.create_regional_comparison_heatmap()
        
        # Step 4: Performance analysis
        self.analyze_performance_correlation()
        
        # Step 5: Comprehensive dashboard
        self.create_comprehensive_dashboard()
        
        # Step 6: Summary report
        self.generate_summary_report()
        
        print("Analysis complete! Check the generated visualizations and report.")
        return self

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EntrepreneurshipNLPAnalyzer('/Users/radman/Desktop/All/projects/ ee-enactus /data/data_cleaned.csv')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis files generated:")
    print("- viz/competency_distribution_heatmap.png")
    print("- viz/program_area_emphasis_heatmap.png") 
    print("- viz/regional_comparison_heatmap.png")
    print("- viz/performance_correlation_heatmap.png")
    print("- viz/comprehensive_dashboard.png")
    print("- viz/entrepreneurship_analysis_report.txt")