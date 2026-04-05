from util.graph_utils import GraphUtils as gu




def main():
        # 1. Define the R-style graph content
        r_content = """
        U_1 -+ X_1, U_1 -+ R_1, U_1 -+ R_2, X_2 -+ R_1,
        X_2 -+ R_2, X_2 -+ Y_2, X_1 -+ R_1, X_1 -+ R_2,
        R_1 -+ A_3, A_3 -+ R_2, A_2 -+ R_2, A_2 -+ E_2,
        R_2 -+ E_2, R_1 -+ A_1, A_1 -+ E_1, E_1 -+ A_4,
        A_4 -+ E_2, E_1 -+ Y_1, E_1 -+ Y_2, E_2 -+ Y_1,
        E_2 -+ Y_2, E_1 -+ E_2
        """

        # 2. Parse the graph using our Utility Class
        print("--- Step 1: Parsing R Formula ---")
        g = gu.from_r_formula(r_content)
        gu.save_graph(g,gu.data_path + 'transitcluster/original/original.dot','dot')
        gu.get_info(g)

if __name__ == "__main__":
    main()