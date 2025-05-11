import logging

from mem0.memory.utils import format_entities

try:
    from langchain_memgraph import Memgraph
except ImportError:
    raise ImportError(
        "langchain_memgraph is not installed. Please install it using pip install langchain-memgraph"
    )

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is not installed. Please install it using pip install rank-bm25"
    )

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = [
    # ——— Molecules & Targets ———
    "compound",  # Candidate peptide or cyclic peptide
    "protein",  # Receptor or ligand protein
    "side_chain",  # Side chain of a protein
    "amino_acid",  # Amino acid
    "target",  # Biological targets (e.g., FcRn, KRAS)
    "mutation",  # Sequence or site mutation (e.g., G12V, I93F)
    "modification",  # Chemical/sequence modification (e.g., cyclization, N-methylation)
    "linker",  # Linker or spacer (e.g., PEG, disulfide)
    "conjugate",  # Conjugates (e.g., fatty acid, PEG, dyes)

    # ——— Experiments & Data ———
    "assay",  # Biological or biophysical assays (e.g., SPR, ELISA)
    "experiment_model",  # Experimental models (e.g., CHO-FcRn, mouse xenograft)
    "sample",  # Biological samples (e.g., plasma, tissue)
    "metric",  # Quantitative results (e.g., KD, IC50, AUC)
    "dataset",  # Dataset name (e.g., KRAS-Peptidome, PDB-Bind)
    "formulation",  # Formulation of a drug

    # ——— Algorithms & AI Systems ———
    "model_architecture",  # Model architecture (e.g., LLM, GNN, Transformer)
    "algorithm",  # Algorithm modules (e.g., scoring, encoder, search)
    "training_strategy",  # Training strategies (e.g., pretraining, distillation)
    "software_tool",  # Tools used (e.g., GraphRAG, FAISS, CrewAI)
    "pipeline",  # Data or model pipeline (ETL, scoring pipeline)

    # ——— Projects & Workflows ———
    "project",  # Drug discovery or AI R&D project name
    "milestone",  # Key development milestones (e.g., IND submission)
    "task",  # Subtasks or assigned work units
    "workflow",  # Internal process or standard operating procedure (SOP)

    # ——— Organizations & Roles ———
    "company",  # Companies (e.g., Novo Nordisk, ChemPartner)
    "organization",  # Research institute, academic lab, or business unit
    "department",  # Internal departments (e.g., Biology, CMC)
    "team",  # Specific teams (e.g., Computational Biology Team)
    "person",  # Named individuals
    "role",  # Functional roles (e.g., PI, data scientist, project manager)

    # ——— Regulatory & Documentation ———
    "guideline",  # Regulatory guidelines (e.g., ICH Q8, FDA draft guidance)
    "regulation",  # Legal regulations (e.g., HIPAA, GDPR)
    "policy",  # Internal or external policies
    "publication",  # Scientific papers, whitepapers
    "patent",  # Intellectual property references

    # ——— General Concepts ———
    "concept",  # Abstract scientific/technical concepts (e.g., immune homeostasis, LLM alignment)
    "technology",  # Enabling tech (e.g., Cryo-EM, AlphaFold)
    "event",  # Scientific events (e.g., AACR 2025, internal review)
    "location"  # Physical locations (labs, cities, countries)
]


class MemoryGraph:

    def __init__(self, config):
        self.config = config

        self.graph = Memgraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = self.config.graph_store.threshold

        # Setup Memgraph:
        # 1. Create vector index (created Entity label on all nodes)
        # 2. Create label property index for performance optimizations
        embedding_dims = self.config.embedder.config["embedding_dims"]
        create_vector_index_query = f"CREATE VECTOR INDEX memzero ON :Entity(embedding) WITH CONFIG {{'dimension': {embedding_dims}, 'capacity': 1000, 'metric': 'cos'}};"
        self.graph.query(create_vector_index_query, params={})
        create_label_prop_index_query = "CREATE INDEX ON :Entity(user_id);"
        self.graph.query(create_label_prop_index_query, params={})
        create_label_index_query = "CREATE INDEX ON :Entity;"
        self.graph.query(create_label_index_query, params={})

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(
            data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(
            entity_type_map.keys()),
                                              filters=filters,
                                              limit=3)

        to_be_deleted = self._get_delete_entities_from_search_output(
            search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted,
                                                 filters["user_id"])
        added_entities = self._add_entities(to_be_added, filters["user_id"],
                                            entity_type_map)

        return {
            "deleted_entities": deleted_entities,
            "added_entities": added_entities
        }

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(
            entity_type_map.keys()),
                                              filters=filters,
                                              limit=10)

        if not search_output:
            return []

        search_outputs_sequence = [[
            item["source"], item["relationship"], item["destination"]
        ] for item in search_output]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query,
                                          search_outputs_sequence,
                                          n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({
                "source": item[0],
                "relationship": item[1],
                "destination": item[2]
            })

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        cypher = """
        MATCH (n {user_id: $user_id})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """

        # return all nodes and relationships
        query = """
        MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity {user_id: $user_id})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query,
                                   params={
                                       "user_id": filters["user_id"],
                                       "limit": limit
                                   })

        final_results = []
        for result in results:
            final_results.append({
                "source": result["source"],
                "relationship": result["relationship"],
                "target": result["target"],
            })

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in [
                "azure_openai_structured", "openai_structured"
        ]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role":
                    "system",
                    "content":
                    ("You are a smart assistant who understands entities and their types in a given text."
                     f"If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity."
                     "Extract all the entities from the text. "
                     "**IMPORTANT:** Extract ONLY entities that belong to the following types or their equivalents/synonyms (e.g., “antibody” -> protein, “SPR assay” -> assay)."
                     f"The following are the allowed entity types: {DEFAULT_ENTITY_TYPES}"
                     "***DO NOT*** answer the question itself if the given text is a question."
                     ),
                },
                {
                    "role": "user",
                    "content": data
                },
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        logger.debug(
            f"Entity type map: {entity_type_map}\n search_results={search_results}"
        )
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters,
                                             entity_type_map):
        """Eshtablish relations among the extracted nodes."""
        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role":
                    "system",
                    "content":
                    EXTRACT_RELATIONS_PROMPT.replace(
                        "USER_ID", filters["user_id"]).replace(
                            "CUSTOM_PROMPT",
                            f"4. {self.config.graph_store.custom_prompt}"),
                },
                {
                    "role": "user",
                    "content": data
                },
            ]
        else:
            messages = [
                {
                    "role":
                    "system",
                    "content":
                    EXTRACT_RELATIONS_PROMPT.replace("USER_ID",
                                                     filters["user_id"]),
                },
                {
                    "role":
                    "user",
                    "content":
                    f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in [
                "azure_openai_structured", "openai_structured"
        ]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities["tool_calls"]:
            entities = extracted_entities["tool_calls"][0]["arguments"][
                "entities"]

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)
            cypher_query = """
            MERGE (q:Query {tmp:true, embedding: $n_embedding})
            WITH q
            MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity)
            WHERE n.embedding IS NOT NULL
            WITH q, COLLECT(n) AS nodes1, COLLECT(m) AS nodes2, r
            UNWIND range(0, size(nodes1)-1) AS idx
            WITH [q] AS q_list, [nodes1[idx]] AS n_list, nodes1[idx] AS n_node, nodes2[idx] AS m_node, r
            CALL node_similarity.cosine_pairwise('embedding', q_list, n_list)
            YIELD node1 AS qnode, node2 AS nnode, similarity
            WITH n_node, m_node, r, similarity
            WHERE similarity >= $threshold
            RETURN n_node.name AS source,
                id(n_node) AS source_id,
                type(r) AS relationship,
                id(r) AS relation_id,
                m_node.name AS destination,
                id(m_node) AS destination_id,
                similarity
            ORDER BY similarity DESC
            LIMIT $limit

            UNION

            // Search for nodes that are connected to the query node
            MERGE (q:Query {tmp:true, embedding: $n_embedding})
            WITH q
            MATCH (n:Entity {user_id: $user_id})<-[r]-(m:Entity)
            WHERE n.embedding IS NOT NULL
            WITH q, COLLECT(n) AS nodes1, COLLECT(m) AS nodes2, r
            UNWIND range(0, size(nodes1)-1) AS idx
            WITH [q] AS q_list, [nodes1[idx]] AS n_list, nodes1[idx] AS n_node, nodes2[idx] AS m_node, r
            CALL node_similarity.cosine_pairwise('embedding', q_list, n_list)
            YIELD node1 AS qnode, node2 AS nnode, similarity
            WITH n_node, m_node, r, similarity
            WHERE similarity >= $threshold
            RETURN m_node.name AS source,
                id(m_node) AS source_id,
                type(r) AS relationship,
                id(r) AS relation_id,
                n_node.name AS destination,
                id(n_node) AS destination_id,
                similarity
            ORDER BY similarity DESC
            LIMIT $limit;
            """
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            ans = self.graph.query(cypher_query, params=params)

            cleanup_query = "MATCH (q:Query {tmp:true}) DETACH DELETE q"
            self.graph.query(cleanup_query, params={})

            result_relations.extend(ans)

        def remove_dups(a):
            seen = set()
            unique = []
            for triplet in a:
                # build a dict without 'similarity'
                filtered = {
                    k: v
                    for k, v in triplet.items() if k != 'similarity'
                }
                # create a hashable signature
                signature = tuple(sorted(filtered.items()))
                if signature not in seen:
                    seen.add(signature)
                    unique.append(filtered)
            return unique

        result_relations = remove_dups(result_relations)
        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data,
                                                filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)
        system_prompt, user_prompt = get_delete_messages(
            search_output_string, data, filters["user_id"])

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in [
                "azure_openai_structured", "openai_structured"
        ]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
            tools=_tools,
        )
        to_be_deleted = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_deleted.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, user_id):
        """Delete the entities from the graph."""
        results = []
        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n:Entity {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m {{name: $dest_name, user_id: $user_id}})
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    # added Entity label to all nodes for vector search to work
    def _add_entities(self, to_be_added, user_id, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        results = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            destination_type = entity_type_map.get(destination, "__User__")

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings; this is basically
            # comparison of one embedding to all embeddings in a graph -> vector
            # search with cosine similarity metric
            source_node_search_result = self._search_source_node(
                source_embedding, user_id, threshold=0.9)
            destination_node_search_result = self._search_destination_node(
                dest_embedding, user_id, threshold=0.9)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                cypher = f"""
                    MATCH (source:Entity)
                    WHERE id(source) = $source_id
                    MERGE (destination:{destination_type}:Entity {{name: $destination_name, user_id: $user_id}})
                    ON CREATE SET
                        destination.created = timestamp(),
                        destination.embedding = $destination_embedding,
                        destination:Entity
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "source_id":
                    source_node_search_result[0]["id(source_candidate)"],
                    "destination_name":
                    destination,
                    "destination_embedding":
                    dest_embedding,
                    "user_id":
                    user_id,
                }
            elif destination_node_search_result and not source_node_search_result:
                cypher = f"""
                    MATCH (destination:Entity)
                    WHERE id(destination) = $destination_id
                    MERGE (source:{source_type}:Entity {{name: $source_name, user_id: $user_id}})
                    ON CREATE SET
                        source.created = timestamp(),
                        source.embedding = $source_embedding,
                        source:Entity
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "destination_id":
                    destination_node_search_result[0]
                    ["id(destination_candidate)"],
                    "source_name":
                    source,
                    "source_embedding":
                    source_embedding,
                    "user_id":
                    user_id,
                }
            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                    MATCH (source:Entity)
                    WHERE id(source) = $source_id
                    MATCH (destination:Entity)
                    WHERE id(destination) = $destination_id
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = timestamp(),
                        r.updated_at = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """
                params = {
                    "source_id":
                    source_node_search_result[0]["id(source_candidate)"],
                    "destination_id":
                    destination_node_search_result[0]
                    ["id(destination_candidate)"],
                    "user_id":
                    user_id,
                }
            else:
                cypher = f"""
                    MERGE (n:{source_type}:Entity {{name: $source_name, user_id: $user_id}})
                    ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding, n:Entity
                    ON MATCH SET n.embedding = $source_embedding
                    MERGE (m:{destination_type}:Entity {{name: $dest_name, user_id: $user_id}})
                    ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding, m:Entity
                    ON MATCH SET m.embedding = $dest_embedding
                    MERGE (n)-[rel:{relationship}]->(m)
                    ON CREATE SET rel.created = timestamp()
                    RETURN n.name AS source, type(rel) AS relationship, m.name AS target
                    """
                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(
                " ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, user_id, threshold=0.9):
        cypher = """
            CALL vector_search.search("memzero", 1, $source_embedding) 
            YIELD distance, node, similarity
            WITH node AS source_candidate, similarity
            WHERE source_candidate.user_id = $user_id AND similarity >= $threshold
            RETURN id(source_candidate);
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": user_id,
            "threshold": threshold,
        }

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self,
                                 destination_embedding,
                                 user_id,
                                 threshold=0.9):
        cypher = """
            CALL vector_search.search("memzero", 1, $destination_embedding) 
            YIELD distance, node, similarity
            WITH node AS destination_candidate, similarity
            WHERE node.user_id = $user_id AND similarity >= $threshold
            RETURN id(destination_candidate);
            """
        params = {
            "destination_embedding": destination_embedding,
            "user_id": user_id,
            "threshold": threshold,
        }

        result = self.graph.query(cypher, params=params)
        return result
