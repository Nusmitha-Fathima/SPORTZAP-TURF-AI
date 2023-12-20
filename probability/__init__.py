from flask import Flask, jsonify,request
import joblib
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)
model = joblib.load('fives_model.joblib')

def get_probdata_from_api():
    api_url = "http://13.126.57.93:8000/admin_app/match_rating_view/"
    response = requests.get(api_url)

    if response.status_code == 200:
        try:
            data = response.json()
            messages = data['message']
            df = pd.DataFrame(messages)
            flattened_players = []
            for entry in messages:
                players = entry['players']
                player_details = {}
                for i, player in enumerate(players, start=1):
                    player_details[f"player{i}_id"] = player[f'player{i}_id']
                    player_details[f"player{i}_skill"] = player[f'player{i}_skill']
                flattened_players.append(player_details)
            df_players = pd.DataFrame(flattened_players)
            df_combined = pd.concat([pd.DataFrame(messages), df_players], axis=1)
            df_combined.drop('players', axis=1, inplace=True)
            player_levels = {}
            for index, row in df_combined.iterrows():
                for i in range(1, 6):
                    player_id = row[f'player{i}_id']
                    player_level = row[f'player{i}_skill']
                    player_levels[player_id] = player_level
            unique_players = set().union(*[set(df_combined[f'player{i}_id']) for i in range(1, 6)])
            player_counts = {player: 0 for player in unique_players}
            player_win_counts = {player: 0 for player in unique_players}
            for i in range(len(df_combined)):
                row = df_combined.iloc[i]
                for player in unique_players:
                    if player in row.values:
                        player_counts[player] += 1
                        if player in row.values and row['result'].lower() == 'win':
                            player_win_counts[player] += 1

            player_counts_df = pd.DataFrame(list(player_counts.items()), columns=['Player', 'Count'])
            player_win_counts_df = pd.DataFrame(list(player_win_counts.items()), columns=['Player', 'Win_Count'])
            player_df = pd.merge(player_counts_df, player_win_counts_df, on='Player', how='outer')
            player_df['win_ratio'] = (player_df['Win_Count'] / player_df['Count']).round(2)
            player_df['Player_Level'] = [player_levels.get(player, 0) for player in player_df['Player']]

            Team_id = set(df_combined['team_id'])
            team_counts = {team: 0 for team in Team_id}
            team_win_counts = {team: 0 for team in Team_id}

            for i in range(len(df_combined)):
                row = df_combined.iloc[i]
                team = row['team_id']
                team_counts[team] += 1
                if row['result'].lower() == 'win':
                    team_win_counts[team] += 1
            team_counts_df = pd.DataFrame(list(team_counts.items()), columns=['Team_ID', 'Count'])
            team_win_counts_df = pd.DataFrame(list(team_win_counts.items()), columns=['Team_ID', 'win_Count'])
            teams_df = pd.merge(team_counts_df, team_win_counts_df, on='Team_ID', how='left')
            teams_df['win_Count'].fillna(0, inplace=True)
            teams_df['win_ratio'] = (teams_df['win_Count'] / teams_df['Count']).round(2)
            return df_combined, player_df, teams_df
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"API request failed with status code: {response.status_code}")
    player_df.to_csv('player_df.csv', index=False)
    return None, None, None


@app.route('/probabilityfives', methods=['POST','GET'])
def probabilityfives():
    try:
        df_combined, player_df, teams_df = get_probdata_from_api()
        if df_combined is None or player_df is None or teams_df is None:
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
        data = request.get_json()
        dic = data.get('teams', [])

        if not dic or not all(isinstance(team, dict) and 'players' in team for team in dic):
            return jsonify({"error": "Invalid or missing team data"}), 400
        list_ = []
        for team_info in dic:
            team_id_key = 'team1' if 'team1' in team_info else 'team2'
            team_id_value = team_info[team_id_key]
            try:
                team_id = int(team_id_value)
            except ValueError:
                print(f"Invalid team_id value: {team_id_value}. It should be an integer.")
                continue  
            matching_rows = teams_df[teams_df['Team_ID'] == team_id]
            print(f"Team ID: {team_id}, Matching Rows:\n{matching_rows}")
            if not matching_rows.empty:
                team_win_ratio = matching_rows['win_ratio'].values[0]
                print(team_win_ratio)
                list_.append(team_win_ratio)
            else:
                print(f"No matching rows found for Team ID: {team_id}")
                list_.append(0)  

            for player in team_info['players']:
                try:
                    player_str = str(player)
                except ValueError:
                    print(f"Invalid player ID value: {player}. It should be convertible to a string.")
                    continue  

                matching_players = player_df[player_df['Player'] == player_str]
                print(f"Player: {player_str}, Matching Players:\n{matching_players}")
                if not matching_players.empty:
                    player_level = matching_players['Player_Level'].values[0]
                    if player_level.lower() == 'beginner':
                        list_.append(0)
                    elif player_level.lower() == 'intermediate':
                        list_.append(1)
                    else:
                        list_.append(2)

                    player_win_ratio = matching_players['win_ratio'].values[0]
                    list_.append(player_win_ratio)

                else:
                    print(f"No matching players found for Player: {player_str}")
                    list_.append(0)
                    list_.append(0)  

        try:
            input_data = np.array(list_).reshape(1, -1)
            probability = model.predict_proba(input_data)
            team_a_probability = round(probability[0, 1] * 100, 2)
            team_b_probability = round(probability[0, 2] * 100, 2)
            draw_probability = round(probability[0, 0] * 100, 2)

        except Exception as e:
            print(f"Error during prediction: {e}")
            team_a_probability = 45.0
            team_b_probability = 45.0
            draw_probability = 10.0

        response = {
            "probability": {
                "Team 1": f"{team_a_probability:.2f}%",
                "Team 2": f"{team_b_probability:.2f}%",
                "Draw": f"{draw_probability:.2f}%"
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/playerdata', methods=['GET'])
def player_data_api():
    df_combined, player_df, teams_df = get_probdata_from_api()

    if player_df is None:
        return jsonify({"error": "Failed to retrieve player data"}), 500

    try:
        player_data_json = player_df.to_json(orient='records')

        return player_data_json
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
