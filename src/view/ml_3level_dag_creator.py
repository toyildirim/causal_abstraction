import networkx as nx
import pickle
import os


class MLGraphGenerator:
    """
    Generic utility to save and load NetworkX Graph objects.
    """

    @staticmethod
    def save_nx_graph(G, filename, folder="graphs"):
        """
        Saves a NetworkX graph object to a binary file (.gpickle).
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, f"{filename}.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(G, f)

        print(f"Graph object saved to: {filepath}")
        return filepath

    @staticmethod
    def load_nx_graph(filepath):
        """
        Loads a NetworkX graph object from a binary file.
        """
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        return G



@staticmethod
def get_grad_student_dag():
    G = nx.DiGraph(name="L3_Graduate_Student")
    edges = [
        ("Bias", "Data"), ("Data", "FeatureExtraction"),
        ("Labels", "Supervised"), ("FeatureExtraction", "Supervised"),
        ("FeatureExtraction", "ManifoldProjection"), ("ProjectionFunction", "ManifoldProjection"),
        ("ManifoldProjection", "DistanceCalculation"), ("DistanceMetric", "DistanceCalculation"),
        ("DistanceCalculation", "Clustering"), ("Clustering", "Unsupervised"),
        ("Actions", "RL"), ("Rewards", "RL"),
        ("Supervised", "Model"), ("Unsupervised", "Model"), ("RL", "Model"),
        ("Hyperparams", "Model"), ("Model", "Training"),
        ("Training", "Prediction"), ("Training", "Interpretability"), ("Bias", "Prediction")
    ]
    G.add_edges_from(edges)
    return G

@staticmethod
def get_teen_dag():
    """Level 2: Functional Meso-Level."""
    G = nx.DiGraph(name="L2_Teenager")
    edges = [
        ("History", "Learning Process"), # History = Data + Bias
        ("Habits", "Learning Process"),
        ("Learning Process", "Algorithm"),
        ("Algorithm", "Output")
    ]
    G.add_edges_from(edges)
    return G

@staticmethod
def get_child_dag():
    """Level 1: Cognitive Macro-Level."""
    G = nx.DiGraph(name="L1_Child")
    edges = [
        ("Examples", "Practice"), # Practice = Training / Learning
        ("Practice", "Patterns"),
        ("Patterns", "Guess")     # Guess = Output
    ]
    G.add_edges_from(edges)
    return G
# --- Usage Example for your ML Hierarchy ---




# ChatGPT 5.3 Created DAGS
@staticmethod
def create_child_level_dag():
    graph = nx.DiGraph()

    nodes = [
        "Examples",
        "ObjectShown",
        "VisibleFeatures",
        "RecognizedPattern",
        "Guess",
        "ActualCategory",
        "Correctness"
    ]

    edges = [
        ("Examples", "RecognizedPattern"),
        ("ObjectShown", "VisibleFeatures"),
        ("VisibleFeatures", "RecognizedPattern"),
        ("RecognizedPattern", "Guess"),
        ("Guess", "Correctness"),
        ("ActualCategory", "Correctness")
    ]

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def create_teen_level_dag():
    graph = nx.DiGraph()

    nodes = [
        "UserData",
        "ContentData",
        "FeatureExtraction",
        "Algorithm",
        "ModelRepresentation",
        "Prediction",
        "Recommendation",
        "UserBehavior",
        "FutureUserData"
    ]

    edges = [
        ("UserData", "FeatureExtraction"),
        ("ContentData", "FeatureExtraction"),
        ("FeatureExtraction", "ModelRepresentation"),
        ("Algorithm", "ModelRepresentation"),
        ("ModelRepresentation", "Prediction"),
        ("Prediction", "Recommendation"),
        ("Recommendation", "UserBehavior"),
        ("UserBehavior", "FutureUserData")
    ]

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def create_graduate_level_dag():
    graph = nx.DiGraph()

    nodes = [
        "TrainingData",
        "Labels",

        "FeatureExtraction",
        "HumanEngineeredFeatures",
        "AutomaticallyLearnedFeatures",

        "SimilarityDistanceStructure",
        "Clusters",

        "SupervisedLearning",
        "UnsupervisedLearning",
        "ReinforcementLearning",

        "ModelType",
        "PredictiveModel",
        "Prediction",
        "Loss",

        "State",
        "Policy",
        "Action",
        "Reward",
        "UpdatedPolicy",

        "InterpretabilityExplanation",
        "UsefulnessForDecision"
    ]

    edges = [
        # feature extraction branch
        ("TrainingData", "FeatureExtraction"),
        ("FeatureExtraction", "HumanEngineeredFeatures"),
        ("FeatureExtraction", "AutomaticallyLearnedFeatures"),

        # unsupervised learning branch
        ("TrainingData", "SimilarityDistanceStructure"),
        ("UnsupervisedLearning", "SimilarityDistanceStructure"),
        ("SimilarityDistanceStructure", "Clusters"),
        ("Clusters", "AutomaticallyLearnedFeatures"),

        # supervised / predictive model branch
        ("HumanEngineeredFeatures", "PredictiveModel"),
        ("AutomaticallyLearnedFeatures", "PredictiveModel"),
        ("Labels", "SupervisedLearning"),
        ("SupervisedLearning", "PredictiveModel"),
        ("ModelType", "PredictiveModel"),

        ("PredictiveModel", "Prediction"),
        ("Prediction", "Loss"),
        ("Labels", "Loss"),

        # reinforcement learning branch
        ("TrainingData", "State"),
        ("ReinforcementLearning", "Policy"),
        ("State", "Policy"),
        ("Policy", "Action"),
        ("Action", "Reward"),
        ("Reward", "UpdatedPolicy"),

        # interpretability and usefulness
        ("PredictiveModel", "InterpretabilityExplanation"),
        ("Prediction", "InterpretabilityExplanation"),
        ("InterpretabilityExplanation", "UsefulnessForDecision"),
        ("Prediction", "UsefulnessForDecision")
    ]

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph

def export_ml_dags():
    util = MLGraphGenerator()

    # 1. Level 3: Graduate Student
    # l3_gpt = create_graduate_level_dag()
    # l3_gemini = get_grad_student_dag()
    # l3_gemini_2 = create_grad_dag();
    l3_gpt_54 = create_grad_level_dag_GPT54();
    l3_gemini_pro = get_graduate_level_dag_gemini_pro()

    # util.save_nx_graph(l3_gpt, "l3_grad_gpt")
    # util.save_nx_graph(l3_gemini, "l3_grad_gemini")
    # util.save_nx_graph(l3_gemini_2, "l3_grad_gemini_2")
    util.save_nx_graph(l3_gpt_54, "l3_grad_gpt_5.4")
    util.save_nx_graph(l3_gemini_pro, "l3_grad_gemini_pro")

    # 2. Level 2: Teenager
    # l2_gpt = create_teen_level_dag()
    # l2_gemini = get_teen_dag()
    # l2_gemini_2 = create_teen_dag()
    l2_gpt_54 = create_teen_level_dag_GPT54()
    l2_gemini_pro = get_teen_level_dag_gemini_pro()

    # util.save_nx_graph(l2_gpt, "l2_teen_gpt")
    # util.save_nx_graph(l2_gemini, "l2_teen_gemini")
    # util.save_nx_graph(l2_gemini_2, "l2_teen_gemini_2")
    util.save_nx_graph(l2_gpt_54, "l2_teen_gpt_5.4")
    util.save_nx_graph(l2_gemini_pro, "l2_teen_gemini_pro")


    # 3. Level 1: Child
    # l1_gpt = create_child_level_dag()
    # l1_gemini = get_child_dag()
    # l1_gemini_2 = create_child_dag()
    l1_gpt_54 = create_child_level_dag_GPT54()
    l1_gemini_pro = get_child_level_dag_gemini_pro()
    # util.save_nx_graph(l1_gpt, "l1_child_gpt")
    # util.save_nx_graph(l1_gemini, "l1_child_gemini")
    # util.save_nx_graph(l1_gemini_2, "l1_child_gemini_2")
    util.save_nx_graph(l1_gpt_54, "l1_child_gpt_5.4")
    util.save_nx_graph(l1_gemini_pro, "l1_child_gemini_pro")



def create_child_dag():
    C = nx.DiGraph()
    # Conceptual nodes
    nodes = [
        ("Examples", {"label": "Looking at Examples"}),
        ("Patterns", {"label": "Identifying Patterns"}),
        ("Guess", {"label": "Making a Guess"})
    ]
    C.add_nodes_from(nodes)

    # Simple Causal Chain
    C.add_edges_from([
        ("Examples", "Patterns"),
        ("Patterns", "Guess")
    ])
    return C


def create_teen_dag():
    T = nx.DiGraph()
    # Functional nodes
    nodes = [
        ("Data_Pool", {"label": "Aggregated Data Examples"}),
        ("Model_Rep", {"label": "Model Representation/Patterns"}),
        ("Prediction", {"label": "Model Guess/Inference"}),
        ("Application", {"label": "User Application (Ads/Recs)"})
    ]
    T.add_nodes_from(nodes)

    # Causal Flow
    T.add_edges_from([
        ("Data_Pool", "Model_Rep"),
        ("Model_Rep", "Prediction"),
        ("Prediction", "Application")
    ])
    return T



def create_grad_dag():
    G = nx.DiGraph()
    # Nodes from the interview: D (Data), P (Practice/Process), O (Output)
    nodes = [
        ("InternetTextDistribution", {"label": "InternetTextDistribution"}),
        ("LexiconsTemplates", {"label": "LexiconsTemplates"}),
        ("AutoExtractedFeatures", {"label": "AutoExtractedFeatures"}),
        ("AlgorithmSelection", {"label": "AlgorithmSelection"}),
        ("FineGrainControl", {"label": "FineGrainControl"}),
        ("TrainingConvergence", {"label": "TrainingConvergence"}),
        ("GrammaticalOutput", {"label": "GrammaticalOutput"}),
        ("MeasuredBias", {"label": "MeasuredBias"})
    ]
    G.add_nodes_from(nodes)

    # Causal Edges
    G.add_edges_from([
        ("InternetTextDistribution", "AutoExtractedFeatures"),
        ("InternetTextDistribution", "AlgorithmSelection"),
        ("LexiconsTemplates", "AlgorithmSelection"),
        ("AutoExtractedFeatures", "AlgorithmSelection"),
        ("AlgorithmSelection", "FineGrainControl"),
        ("FineGrainControl", "TrainingConvergence"),
        ("TrainingConvergence", "GrammaticalOutput"),
        ("TrainingConvergence", "MeasuredBias")
    ])
    return G


#ChatGPT 5.4 Thinking

EXPOSURE = "CollectedData"
OUTCOME = "RealWorldImpact"
# EXPOSURE = "exposure"
# OUTCOME = "outcome"

def create_grad_level_dag_GPT54():
    """
    Create the Grad-level DAG.
    Lowest abstraction.

    Exposure and outcome are strict/constant:
        Exposure = Collected data
        Outcome = Real-world impact
    """
    nodes = [
        EXPOSURE,
        OUTCOME,

        "RealWorldProcess",
        "SocialBiasMeasurementBias",
        "DataQuality",
        "Representativeness",
        "Labels",
        "Preprocessing",
        "LearningSetup",
        "SupervisedLearning",
        "UnsupervisedLearning",
        "ReinforcementLearning",
        "DeepLearning",
        "FeatureEngineeringRepresentationLearning",
        "ModelArchitecture",
        "TrainingObjective",
        "OptimizationParameterLearning",
        "TrainedModel",
        "Prediction",
        "AccuracyEvaluation",
        "UsefulnessEvaluation",
        "InterpretabilityEvaluation",
        "FairnessBiasEvaluation",
        "DeploymentDecision",
        "AutomatedDecisionProductBehavior",
    ]

    edges = [
        ("RealWorldProcess", EXPOSURE),
        ("RealWorldProcess", "SocialBiasMeasurementBias"),
        ("SocialBiasMeasurementBias", EXPOSURE),

        (EXPOSURE, "DataQuality"),
        (EXPOSURE, "Representativeness"),
        (EXPOSURE, "Labels"),

        ("DataQuality", "Preprocessing"),
        ("Representativeness", "Preprocessing"),
        ("Labels", "LearningSetup"),

        ("LearningSetup", "SupervisedLearning"),
        ("LearningSetup", "UnsupervisedLearning"),
        ("LearningSetup", "ReinforcementLearning"),
        ("LearningSetup", "DeepLearning"),

        ("Preprocessing", "FeatureEngineeringRepresentationLearning"),

        ("FeatureEngineeringRepresentationLearning", "ModelArchitecture"),
        ("SupervisedLearning", "ModelArchitecture"),
        ("UnsupervisedLearning", "ModelArchitecture"),
        ("ReinforcementLearning", "ModelArchitecture"),
        ("DeepLearning", "ModelArchitecture"),

        ("ModelArchitecture", "TrainingObjective"),
        ("TrainingObjective", "OptimizationParameterLearning"),
        ("OptimizationParameterLearning", "TrainedModel"),

        ("TrainedModel", "Prediction"),

        ("Prediction", "AccuracyEvaluation"),
        ("Prediction", "UsefulnessEvaluation"),
        ("Prediction", "InterpretabilityEvaluation"),
        ("Prediction", "FairnessBiasEvaluation"),

        ("AccuracyEvaluation", "DeploymentDecision"),
        ("UsefulnessEvaluation", "DeploymentDecision"),
        ("InterpretabilityEvaluation", "DeploymentDecision"),
        ("FairnessBiasEvaluation", "DeploymentDecision"),

        ("DeploymentDecision", "AutomatedDecisionProductBehavior"),
        ("AutomatedDecisionProductBehavior", OUTCOME),
    ]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    G.graph["name"] = "Grad-level DAG — Lowest Abstraction"
    G.graph["level"] = "grad"
    # G.graph["exposure"] = EXPOSURE
    # G.graph["outcome"] = OUTCOME
    G.nodes[EXPOSURE]["label"] = "exposure"
    G.nodes[EXPOSURE]["role"] = "exposure"

    G.nodes[OUTCOME]["label"] = "outcome"
    G.nodes[OUTCOME]["role"] = "outcome"

    assert nx.is_directed_acyclic_graph(G), "Grad-level graph is not a DAG."
    assert EXPOSURE in G.nodes, "Exposure node missing from Grad-level DAG."
    assert OUTCOME in G.nodes, "Outcome node missing from Grad-level DAG."

    return G


def create_teen_level_dag_GPT54():
    """
    Create the Teen-level DAG.
    Middle abstraction.

    Exposure and outcome are strict/constant:
        Exposure = Collected data
        Outcome = Real-world impact

    Therefore, Collected data and Real-world impact are not merged.
    """
    nodes = [
        EXPOSURE,
        OUTCOME,

        "RealWorldProcess_SocialBiasMeasurementBias",
        "DataQuality_Representativeness_Labels",
        "Preprocessing_FeatureEngineeringRepresentationLearning",
        "LearningSetup_SupervisedLearning_UnsupervisedLearning_ReinforcementLearning_DeepLearning_ModelArchitecture",
        "TrainingObjective_OptimizationParameterLearning",
        "TrainedModel",
        "Prediction",
        "AccuracyEvaluation_UsefulnessEvaluation_InterpretabilityEvaluation_FairnessBiasEvaluation",
        "DeploymentDecision_AutomatedDecisionProductBehavior",
    ]

    edges = [
        (
            "RealWorldProcess_SocialBiasMeasurementBias",
            EXPOSURE,
        ),
        (
            EXPOSURE,
            "DataQuality_Representativeness_Labels",
        ),
        (
            "DataQuality_Representativeness_Labels",
            "Preprocessing_FeatureEngineeringRepresentationLearning",
        ),
        (
            "Preprocessing_FeatureEngineeringRepresentationLearning",
            "LearningSetup_SupervisedLearning_UnsupervisedLearning_ReinforcementLearning_DeepLearning_ModelArchitecture",
        ),
        (
            "LearningSetup_SupervisedLearning_UnsupervisedLearning_ReinforcementLearning_DeepLearning_ModelArchitecture",
            "TrainingObjective_OptimizationParameterLearning",
        ),
        (
            "TrainingObjective_OptimizationParameterLearning",
            "TrainedModel",
        ),
        (
            "TrainedModel",
            "Prediction",
        ),
        (
            "Prediction",
            "AccuracyEvaluation_UsefulnessEvaluation_InterpretabilityEvaluation_FairnessBiasEvaluation",
        ),
        (
            "AccuracyEvaluation_UsefulnessEvaluation_InterpretabilityEvaluation_FairnessBiasEvaluation",
            "DeploymentDecision_AutomatedDecisionProductBehavior",
        ),
        (
            "DeploymentDecision_AutomatedDecisionProductBehavior",
            OUTCOME,
        ),
    ]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    G.graph["name"] = "Teen-level DAG — Middle Abstraction"
    G.graph["level"] = "teen"
    # G.graph["exposure"] = EXPOSURE
    # G.graph["outcome"] = OUTCOME
    G.nodes[EXPOSURE]["label"] = "exposure"
    G.nodes[EXPOSURE]["role"] = "exposure"

    G.nodes[OUTCOME]["label"] = "outcome"
    G.nodes[OUTCOME]["role"] = "outcome"

    assert nx.is_directed_acyclic_graph(G), "Teen-level graph is not a DAG."
    assert EXPOSURE in G.nodes, "Exposure node missing from Teen-level DAG."
    assert OUTCOME in G.nodes, "Outcome node missing from Teen-level DAG."

    return G


def create_child_level_dag_GPT54():
    """
    Create the Child-level DAG.
    Highest abstraction.

    Exposure and outcome are strict/constant:
        Exposure = Collected data
        Outcome = Real-world impact

    Therefore, Collected data and Real-world impact are not merged.
    """
    nodes = [
        EXPOSURE,
        OUTCOME,

        "RealWorldProcess_SocialBiasMeasurementBias",
        "DataQuality_Representativeness_Labels_Preprocessing_FeatureEngineeringRepresentationLearning",
        "LearningSetup_SupervisedLearning_UnsupervisedLearning_ReinforcementLearning_DeepLearning_ModelArchitecture_TrainingObjective_OptimizationParameterLearning_TrainedModel",
        "Prediction",
        "AccuracyEvaluation_UsefulnessEvaluation_InterpretabilityEvaluation_FairnessBiasEvaluation_DeploymentDecision_AutomatedDecisionProductBehavior",
    ]

    edges = [
        (
            "RealWorldProcess_SocialBiasMeasurementBias",
            EXPOSURE,
        ),
        (
            EXPOSURE,
            "DataQuality_Representativeness_Labels_Preprocessing_FeatureEngineeringRepresentationLearning",
        ),
        (
            "DataQuality_Representativeness_Labels_Preprocessing_FeatureEngineeringRepresentationLearning",
            "LearningSetup_SupervisedLearning_UnsupervisedLearning_ReinforcementLearning_DeepLearning_ModelArchitecture_TrainingObjective_OptimizationParameterLearning_TrainedModel",
        ),
        (
            "LearningSetup_SupervisedLearning_UnsupervisedLearning_ReinforcementLearning_DeepLearning_ModelArchitecture_TrainingObjective_OptimizationParameterLearning_TrainedModel",
            "Prediction",
        ),
        (
            "Prediction",
            "AccuracyEvaluation_UsefulnessEvaluation_InterpretabilityEvaluation_FairnessBiasEvaluation_DeploymentDecision_AutomatedDecisionProductBehavior",
        ),
        (
            "AccuracyEvaluation_UsefulnessEvaluation_InterpretabilityEvaluation_FairnessBiasEvaluation_DeploymentDecision_AutomatedDecisionProductBehavior",
            OUTCOME,
        ),
    ]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    G.graph["name"] = "Child-level DAG — Highest Abstraction"
    G.graph["level"] = "child"
    # G.graph["exposure"] = EXPOSURE
    # G.graph["outcome"] = OUTCOME
    G.nodes[EXPOSURE]["label"] = "exposure"
    G.nodes[EXPOSURE]["role"] = "exposure"

    G.nodes[OUTCOME]["label"] = "outcome"
    G.nodes[OUTCOME]["role"] = "outcome"
    assert nx.is_directed_acyclic_graph(G), "Child-level graph is not a DAG."
    assert EXPOSURE in G.nodes, "Exposure node missing from Child-level DAG."
    assert OUTCOME in G.nodes, "Outcome node missing from Child-level DAG."

    return G


def get_exposure_outcome(G):
    """
    Return the strict/constant exposure and outcome for a DAG.
    """
    return G.graph["exposure"], G.graph["outcome"]


def get_child_level_dag_gemini_pro():
    """
    Highest Abstraction: All intermediate nodes are aggregated into a single unit.
    Exposure: Data | Outcome: Prediction
    """
    G = nx.DiGraph()
    # Exposure and Outcome are protected
    # Intermediates: Feature Engineering + Labels + Supervised Training + Model Architecture + Model
    aggregated_node = "Feature Engineering + Labels + Supervised Training + Model Architecture + Model"

    # G.add_edge("Data", aggregated_node)
    G.add_edge("Exposure", aggregated_node)
    # G.add_edge(aggregated_node, "Prediction")
    G.add_edge(aggregated_node, "Outcome")
    return G


def get_teen_level_dag_gemini_pro():
    """
    Mid-Level Abstraction: Nodes are grouped into 'Patterns' and 'Algorithms'.
    Exposure: Data | Outcome: Prediction
    """
    G = nx.DiGraph()
    # Grouping based on the interview's description of 'Patterns' and 'Algorithms' [cite: 19, 78]
    patterns_node = "Feature Engineering + Labels"
    algorithm_node = "Supervised Training + Model Architecture + Model"

    # G.add_edge("Data", patterns_node)
    G.add_edge("Exposure", patterns_node)
    G.add_edge(patterns_node, algorithm_node)
    # G.add_edge(algorithm_node, "Prediction")
    G.add_edge(algorithm_node, "Outcome")

    return G


def get_graduate_level_dag_gemini_pro():
    """
    Lowest Abstraction: Full granularity as described by the PhD student and Hilary.
    Exposure: Data | Outcome: Prediction
    """
    G = nx.DiGraph()
    # Full one-to-one mapping of technical nodes [cite: 102, 110, 138]
    # G.add_edge("Data", "Feature Engineering")
    # G.add_edge("Data", "Labels")
    G.add_edge("Exposure", "Feature Engineering")
    G.add_edge("Exposure", "Labels")
    G.add_edge("Feature Engineering", "Supervised Training")
    G.add_edge("Labels", "Supervised Training")
    G.add_edge("Model Architecture", "Supervised Training")
    G.add_edge("Supervised Training", "Model")
    # G.add_edge("Model", "Prediction")
    G.add_edge("Model", "Outcome")
    return G


# Example of how to visualize any of these:
# dag = get_graduate_level_dag()
# nx.draw(dag, with_labels=True)
# plt.show()

if __name__ == "__main__":
    export_ml_dags()