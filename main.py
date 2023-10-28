import csv
import heapq
from datetime import datetime, timedelta
INF = 10000000
from collections import defaultdict, deque


class TrainConnection:
    def __init__(self, train_no, islno, station_code,source_station_code, arrival_time, departure_time, distance):
        self.train_no = train_no
        self.islno = islno
        self.station_code = station_code
        self.source_station_code = source_station_code
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.distance = distance

def read_train_schedule(file_path):
    train_schedule = []
    # Read the train schedule data from the CSV file and store it in the train_schedule dictionary
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            train_no = row[0]
            islno = row[2]
            station_code = row[3]
            station_code= station_code.replace(" ","")
            source_station_code= row[8]
            source_station_code = source_station_code.replace(" ", "")
            arrival_time = row[5]
            departure_time = row[6]
            distance = int(row[7])
            connection = TrainConnection(train_no, islno, station_code, source_station_code, arrival_time, departure_time, distance)
        #    print(connection.train_no, " ", connection.islno, ":", connection.station_code)
            #if station_code in train_schedule:
            train_schedule.append(connection)
            #print(train_schedule)
            #else:
            #    train_schedule[station_code] = [connection]
            #print(train_schedule[station_code])
    return train_schedule

class DirectedGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(deque)
        self.distances = {}
        self.journey = {}
        self.train_number = set()

    def addNode(self, value):
        self.nodes.add(value)

    def addEdge(self, fromNode, toNode, distance, isl_no, train_number, arrival_time, departure_time):
        self.edges[fromNode].append(toNode)
        self.distances[(fromNode, toNode)] = distance
        self.train_number.add(train_number)
        self.journey[(fromNode, toNode)] = [isl_no, train_number, arrival_time, departure_time]


def dijkstra(graph, start, destination, schedule_file, cost_function):
    visited = set()
    distances = {node: (INF, "", "", "", "", "") for node in graph.nodes}
    distances[start] = (0, "", "", "", "", "")
    queue = [(0, start, "", "", "", "")]
    start_arrival_time = None
    start_departure_time = None
    while queue:
        current_distance, current_node, current_train, current_isl_no, current_arrival_time, current_departure_time = heapq.heappop(queue)

        if current_node in visited:
            continue

        visited.add(current_node)
        # If the destination is reached, no need to search further
        if current_node == destination:
            break

        for neighbor in graph.edges[current_node]:
            distance = distances[current_node][0] + graph.distances[(current_node, neighbor)]
            isl, train, arrival_time, departure_time = graph.journey[(current_node, neighbor)]
            train_no = train
            isl_no = isl

            if distance < distances[neighbor][0]:
                distances[neighbor] = (distance, current_node, train_no, isl_no, arrival_time, departure_time)
                heapq.heappush(queue, (distance, neighbor, train_no, isl_no, arrival_time, departure_time))

    print(visited)

    current_node = destination
    # finding the start node arrival and departure time for further calculations of arrival time cost

    while current_node != start:
        next_node = distances[current_node][1]
        if distances[next_node][3] == "":
            print("last stop reached")
            identified_train = distances[current_node][2]
            if schedule_file == "mini-schedule.csv":
                for connection in train_schedule:
                    if connection.station_code == start and connection.train_no == identified_train:
                        start_arrival_time = connection.arrival_time
                        start_departure_time = connection.departure_time
                        break
            elif schedule_file == "schedule.csv":
                for connect in train_schedule_problems_csv:
                    if connect.station_code == start and connect.train_no == identified_train:
                        start_arrival_time = connect.arrival_time
                        start_departure_time = connect.departure_time
                        break

            if start_arrival_time is None or start_departure_time is None:
                raise ValueError(
                    f"Arrival and departure times for the start node '{start}' not found in the schedule data.")
        current_node = next_node

    if cost_function == "price":
        result, distance, price, destination_arrival_time, days, start_departure_time = \
            calculate_price(destination, distances, start, start_arrival_time, start_departure_time)
    else:
        current_node = destination
        result = []
        days = 0
        price = 0
        distance = distances[current_node][0]
        first_train_no = distances[destination][2]
        first_station = distances[current_node][3]
        destination_arrival_time = distances[current_node][4]
        last_stop_arrival_time = distances[current_node][4]
        last_stop_departure_time = distances[current_node][5]
        train_arrival_time = distances[current_node][4]
        train_departure_time = distances[current_node][5]
        print(first_station)
        previous_station = first_station
        first_count = 0
        while current_node != start:
            next_node = distances[current_node][1]
            train_no = distances[current_node][2]
            isl_no = distances[current_node][3]
            train_arrival_time = distances[current_node][4]
            train_departure_time = distances[current_node][5]
            print("current train:", train_no, "previous train:", first_train_no, "current station:", isl_no,
                  "previous station:", previous_station)
            if train_no == first_train_no:
                first_count += 1
                if distances[next_node][3] == "":
                    if isl_no != previous_station:
                        if train_arrival_time < start_departure_time:
                            print("1000000000000")
                            days += 1
                        if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                            if last_stop_arrival_time < train_departure_time:
                                print("200000000000")
                                days += 1
                            if train_departure_time < train_arrival_time and first_count > 1:
                                print("30000000000")
                                days += 1
                    result.insert(0, (first_train_no, int(isl_no) - 1, first_station))
                else:
                    if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                        # In general we will always need to “add a day” if the
                        # arrival time at stop n is smaller than the departure time at stop n − 1. We will do the same
                        # for the departure time at any stop if it is smaller than the arrival time, which is relevant for
                        # some of the cost functions.
                        print("first train:", first_train_no)
                        print("current train:", train_no)
                        print("check:", train_departure_time, last_stop_arrival_time)
                        if last_stop_arrival_time < train_departure_time:
                            days += 1
                            print("day added")
                        print("first count", first_count)
                        if train_departure_time < train_arrival_time and first_count > 1:
                            print("adding the day")
                            days += 1
                # '15111': 2 -> 13;'14203': 4 -> 5;'14266': 42 -> 42;'14204': 9 -> 10;'14266': 38 -> 39;'54253': 14 -> 16,373
            else:
                if distances[next_node][3] == "":
                    if train_arrival_time < start_departure_time:
                        print("5555555")
                        days += 1
                    # if train_departure_time < start_arrival_time:
                    #    print("<6666666>")
                    #    days += 1
                    #    price += 1
                    if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                        train_arrive_time = None
                        current_departure_time = None
                        current_arrival_time = None
                        if schedule_file == "mini-schedule.csv":
                            for connection in train_schedule:
                                if connection.station_code == current_node and connection.train_no == train_no:
                                    train_arrive_time = connection.arrival_time
                                if connection.station_code == current_node and connection.islno == str(int(previous_station) - 1) and connection.train_no == first_train_no:
                                    current_arrival_time = connection.arrival_time
                                    current_departure_time = connection.departure_time
                        elif schedule_file == "schedule.csv":
                            for connect in train_schedule_problems_csv:
                                if connect.station_code == current_node and connect.train_no == train_no:
                                    train_arrive_time = connect.arrival_time
                                if connect.station_code == current_node and connect.islno == str(int(previous_station) - 1) and connect.train_no == first_train_no:
                                    current_arrival_time = connect.arrival_time
                                    current_departure_time = connect.departure_time

                        if current_departure_time != None and train_arrive_time != None:
                            if current_departure_time < train_arrive_time:
                                days += 1
                                print("77777777")
                    result.insert(0, (first_train_no, int(previous_station) - 1, first_station))
                    result.insert(0, (train_no, int(isl_no) - 1, isl_no))
                else:
                    flag = False
                    if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                        train_arrive_time = None
                        current_departure_time = None
                        current_arrival_time = None
                        current_train_minus_one_departs_at = None
                        train_depart_time = None
                        current_train_minus_one_arrives_at = None
                        print("current node", current_node)
                        if schedule_file == "mini-schedule.csv":
                            for connection in train_schedule:
                                if connection.station_code == current_node and connection.train_no == train_no:
                                    train_arrive_time = connection.arrival_time
                                    train_depart_time = connection.departure_time

                                if connection.station_code == current_node and connection.islno == \
                                        str(int(previous_station) - 1) and connection.train_no == first_train_no:
                                    current_train_minus_one_departs_at = connection.departure_time
                                    current_train_minus_one_arrives_at = connection.arrival_time

                                if connection.station_code == current_node and connection.train_no == first_train_no:
                                    current_arrival_time = connection.arrival_time
                                    current_departure_time = connection.departure_time
                        elif schedule_file == "schedule.csv":
                            for connect in train_schedule_problems_csv:
                                if connect.station_code == current_node and connect.train_no == train_no:
                                    train_arrive_time = connect.arrival_time
                                    train_depart_time = connect.departure_time

                                if connect.station_code == current_node and connect.islno == \
                                        str(int(previous_station) - 1) and connect.train_no == first_train_no:
                                    current_train_minus_one_departs_at = connect.departure_time
                                    current_train_minus_one_arrives_at = connect.arrival_time

                                if connect.station_code == current_node and connect.train_no == first_train_no:
                                    current_arrival_time = connect.arrival_time
                                    current_departure_time = connect.departure_time

                        if current_train_minus_one_departs_at != None and train_arrive_time != None:
                            print("train_arrive_time222:", train_arrive_time)
                            print("current_train_minus_one_departs_at", current_train_minus_one_departs_at)

                            #                        if train_arrive_time < current_train_minus_one_departs_at:
                            #                            if s == int(isl_no):
                            #                                print("current train minus 1 price added")
                            #                                days += 1
                            #                                price += 1
                            if last_stop_arrival_time < current_train_minus_one_departs_at:
                                days += 1
                                print("11111111")

                        if current_train_minus_one_departs_at != None and current_train_minus_one_arrives_at != None:
                            last_stop_departure_time = current_train_minus_one_departs_at
                            last_stop_arrival_time = current_train_minus_one_arrives_at

                        s = int(previous_station) - 1
                        print(isl_no)
                        print("previous_station)-1", s)

                        if current_departure_time is not None and train_arrive_time is not None:
                            print("train_arrive_time:888888", train_arrive_time)
                            print("current_train_minus_one_departs_at", current_train_minus_one_departs_at)

                            #                        if train_arrive_time < current_train_minus_one_departs_at:
                            #                            if s == int(isl_no):
                            #                                print("current train minus 1 price added")
                            #                                days += 1
                            #                                price += 1
                            if current_departure_time < train_arrive_time:
                                days += 1
                                print("888888888")
                        print("train_departure_time", train_departure_time, "train_arrival_time", train_arrival_time)
                        # if train_departure_time < train_arrival_time and first_count > 1:
                        #    print("1010101010")
                        #    days += 1
                        train_arrival_time = train_arrival_time.replace("'", "")

                        last_stop_departure_time = last_stop_departure_time.replace("'", "")
                        print("last_stop_departure_time:", last_stop_departure_time)
                        previous_arrival_time = datetime.strptime(train_arrival_time, "%H:%M:%S")

                        next_departure_time = datetime.strptime(last_stop_departure_time, "%H:%M:%S")

                        if next_departure_time < previous_arrival_time:
                            next_departure_time += timedelta(days=1)

                        changeover_time = timedelta(minutes=10)
                        # train change requires 10 minutes minimum
                        new_departure_time = previous_arrival_time + changeover_time
                        print("new_departure_time:", new_departure_time)
                        print("next_departure_time:", next_departure_time)

                        time_difference = next_departure_time - new_departure_time
                        print(f"Time Difference: {time_difference}")
                        if time_difference < timedelta(0):
                            print("The time difference is negative.")
                            days += 1
                            print("999999999999")
                    print("inserting:", first_train_no, " Train from: ", int(previous_station) - 1, "to: ",
                          first_station)
                    result.insert(0, (first_train_no, int(previous_station) - 1, first_station))
                    first_count = 0
                first_train_no = train_no
                first_station = isl_no
            previous_station = isl_no
            last_stop_arrival_time = distances[current_node][4]
            last_stop_departure_time = distances[current_node][5]
            current_node = next_node
    return result, distance, price, destination_arrival_time, days, start_departure_time


def calculate_price(destination, distances, start, start_arrival_time, start_departure_time):
    current_node = destination
    result = []
    days = 0
    distance = distances[current_node][0]
    first_train_no = distances[destination][2]
    first_station = distances[current_node][3]
    destination_arrival_time = distances[current_node][4]
    last_stop_arrival_time = distances[current_node][4]
    last_stop_departure_time = distances[current_node][5]
    train_arrival_time = distances[current_node][4]
    train_departure_time = distances[current_node][5]
    price = 1
    previous_station = first_station
    first_count = 0
    while current_node != start:
        next_node = distances[current_node][1]
        train_no = distances[current_node][2]
        isl_no = distances[current_node][3]
        train_arrival_time = distances[current_node][4]
        train_departure_time = distances[current_node][5]
        print("current train:", train_no, "previous train:", first_train_no, "current station:", isl_no, "previous station:",previous_station)
        if train_no == first_train_no:
            first_count += 1
            if distances[next_node][3] == "":
                if isl_no != previous_station:
                    if train_arrival_time < start_departure_time:
                        print("1000000000000")
                        days += 1
                        price += 1
                    if last_stop_arrival_time < train_departure_time:
                        print("200000000000")
                        days += 1
                        price += 1
                    if last_stop_departure_time < last_stop_arrival_time and first_count > 1:
                        print("30000000000")
                        days += 1
                        price += 1
                result.insert(0, (first_train_no, int(isl_no) - 1, first_station))
            else:
                if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                    # In general we will always need to “add a day” if the
                    # arrival time at stop n is smaller than the departure time at stop n − 1. We will do the same
                    # for the departure time at any stop if it is smaller than the arrival time, which is relevant for
                    # some of the cost functions.
                    print("first train:", first_train_no)
                    print("current train:", train_no)
                    print("check:", last_stop_arrival_time, train_departure_time)
                    if last_stop_arrival_time < train_departure_time:
                        days += 1
                        price += 1
                        print("day added")
                    if last_stop_departure_time < last_stop_arrival_time and first_count > 1:
                        print("adding the price")
                        days += 1
                        price += 1
            # '15111': 2 -> 13;'14203': 4 -> 5;'14266': 42 -> 42;'14204': 9 -> 10;'14266': 38 -> 39;'54253': 14 -> 16,373
        else:
            price += 1
            if distances[next_node][3] == "":
                if train_arrival_time < start_departure_time:
                    days += 1
                    price += 1
                if train_departure_time < start_arrival_time:
                    days += 1
                    price += 1
                if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                    train_arrive_time = None
                    current_departure_time = None
                    current_arrival_time = None
                    for connection in train_schedule:
                        if connection.station_code == current_node and connection.train_no == train_no:
                            train_arrive_time = connection.arrival_time
                        if connection.station_code == current_node and connection.train_no == first_train_no:
                            current_arrival_time = connection.arrival_time
                            current_departure_time = connection.departure_time

                    if current_departure_time != None and train_arrive_time != None:
                        if current_departure_time < train_arrive_time:
                            days += 1
                result.insert(0, (first_train_no, int(previous_station) - 1, first_station))
                result.insert(0, (train_no, int(isl_no) - 1, isl_no))
            else:
                flag = False
                if last_stop_departure_time != train_departure_time and last_stop_arrival_time != train_arrival_time:
                    train_arrive_time = None
                    current_departure_time = None
                    current_arrival_time = None
                    current_train_minus_one_departs_at= None
                    train_depart_time = None
                    for connection in train_schedule:
                        if connection.station_code == current_node and connection.train_no == train_no:
                            train_arrive_time = connection.arrival_time
                            train_depart_time = connection.departure_time

                        if connection.station_code == current_node and connection.islno == str(int(previous_station)-1) and connection.train_no == train_no:
                            current_train_minus_one_departs_at = connection.departure_time

                        if connection.station_code == current_node and connection.train_no == first_train_no:
                            current_arrival_time = connection.arrival_time
                            current_departure_time = connection.departure_time
                    s = int(previous_station)-1
                    print(isl_no)
                    print("previous_station)-1", s)
                    if current_departure_time != None and train_arrive_time != None and current_train_minus_one_departs_at!=None:
                        print("train_arrive_time:", train_arrive_time)
                        print("current_train_minus_one_departs_at", current_train_minus_one_departs_at)

#                        if train_arrive_time < current_train_minus_one_departs_at:
#                            if s == int(isl_no):
#                                print("current train minus 1 price added")
#                                days += 1
#                                price += 1
                        if current_departure_time < train_arrive_time:
                            days += 1
                    if last_stop_departure_time < last_stop_arrival_time and first_count > 0:
                        days += 1
                        price += 1
                        print("22487 fixed")
                    train_arrival_time = train_arrival_time.replace("'", "")

                    last_stop_departure_time = last_stop_departure_time.replace("'", "")

                    previous_arrival_time = datetime.strptime(train_arrival_time, "%H:%M:%S")

                    next_departure_time = datetime.strptime(last_stop_departure_time, "%H:%M:%S")

                    changeover_time = timedelta(minutes=10)
                        # train change requires 10 minutes minimum
                    new_departure_time = previous_arrival_time + changeover_time
                    time_difference = next_departure_time - new_departure_time
                    print(f"Time Difference: {time_difference}")
                    if time_difference < timedelta(0):
                        print("The time difference is negative.")
                        days += 1
                print("inserting:", first_train_no, " Train from: ", int(previous_station) - 1, "to: ", first_station,
                      "price", price)
                result.insert(0, (first_train_no, int(previous_station) - 1, first_station))
                first_count = 0
            first_train_no = train_no
            first_station = isl_no
        previous_station = isl_no
        last_stop_arrival_time = distances[current_node][4]
        last_stop_departure_time = distances[current_node][5]
        current_node = next_node
    return result, distance, price, destination_arrival_time, days, start_departure_time


def replace_last_occurrence(input_string, search_string, replace_string):
    last_index = input_string.rfind(search_string)

    if last_index != -1:
        modified_string = input_string[:last_index] + replace_string + input_string[
                                                                       last_index + len(search_string):]
        return modified_string
    else:
        return input_string


class Problems_det:
    def __init__(self, ProblemNo, FromStation, ToStation, Schedule, CostFunction):
        self.ProblemNo = ProblemNo
        self.FromStation = FromStation
        self.ToStation = ToStation
        self.Schedule = Schedule
        self.CostFunction = CostFunction


def read_problems_from_file(file_path):
    problems = {}

    # Read the train schedule data from the CSV file and store it in a dictionary
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            ProblemNo = row[0]
            FromStation = row[1]
            ToStation = row[2]
            Schedule = row[3]
            CostFunction = row[4]

            problem_details = Problems_det(ProblemNo, FromStation, ToStation, Schedule, CostFunction)

            problems[ProblemNo] = problem_details

    return problems


def write_to_file(result,problem_no, final_solution_file):
    f = open(final_solution_file, "a")
    final_result= problem_no+result+"\n"
    f.write(final_result)
    f.close()


if __name__ == "__main__":
    train_schedule = read_train_schedule("mini-schedule.csv")
    problems = read_problems_from_file("problems.csv")
    #problems = read_problems_from_file("example-problems.csv")

    # problems = read_problems_from_file("problems.csv")

    train_schedule_problems_csv = read_train_schedule("schedule.csv")
    #final_solution_file = "solutions_example.csv"
    final_solution_file = "solutions.csv"
    train_numbers = set()
    customGraph = DirectedGraph()
    customGraph_problem_csv = DirectedGraph()

    file = open(final_solution_file, "w")
    file.truncate()
    # Close the file
    file.close()


    f = open(final_solution_file, "a")
    final_result = "ProblemNo,Connection,Cost"+"\n"
    f.write(final_result)
    f.close()

#schedule.csv file graph development
    for i in train_schedule_problems_csv:
        train_numbers.add(i.train_no)

    prev_station = ""
    prev_distance = 0

    for i in train_schedule_problems_csv:
        if i.train_no in train_numbers:
            customGraph_problem_csv.addNode(i.station_code)
            toStation = i.station_code
            distance = i.distance
            if int(i.islno) == 1:
                fromstation = i.source_station_code
                toStation = i.station_code
                distance = i.distance
                prev_station = toStation
                prev_distance = distance
                distance_from_source_station = 0
                customGraph_problem_csv.addEdge(fromstation, toStation, distance, i.islno, i.train_no, i.arrival_time,
                                    i.departure_time)
            elif int(i.islno) > 1:
                fromstation = prev_station
                distance = distance - prev_distance
                customGraph_problem_csv.addEdge(fromstation, toStation, distance, i.islno, i.train_no, i.arrival_time,
                                    i.departure_time)
                prev_station = toStation
                prev_distance = i.distance
#Mini schedule file graph development
    for i in train_schedule:
        train_numbers.add(i.train_no)

    prev_station = ""
    prev_distance = 0

    for i in train_schedule:
        if i.train_no in train_numbers:
            customGraph.addNode(i.station_code)
            toStation = i.station_code
            distance = i.distance
            if int(i.islno) == 1:
                fromstation = i.source_station_code
                toStation = i.station_code
                distance = i.distance
                prev_station = toStation
                prev_distance = distance
                distance_from_source_station = 0
                customGraph.addEdge(fromstation, toStation, distance, i.islno, i.train_no, i.arrival_time, i.departure_time)
            elif int(i.islno) > 1:
                fromstation = prev_station
                distance = distance - prev_distance
                customGraph.addEdge(fromstation, toStation, distance, i.islno, i.train_no, i.arrival_time, i.departure_time)
                prev_station = toStation
                prev_distance = i.distance
#JNU,CDMR-(error)  GMO,DHN  BLM,KVG MLPR,RNG BNVD,BLM
    for problem in problems:
        price = None
        print("problem: ", problems[problem].ProblemNo, problems[problem].FromStation, problems[problem].ToStation,
              problems[problem].Schedule, problems[problem].CostFunction)
        from_station = problems[problem].FromStation
        to_station = problems[problem].ToStation
        #distance, stops
        cost_function = problems[problem].CostFunction

        schedule_file = problems[problem].Schedule
        cost_function = cost_function.split()
        if schedule_file == "mini-schedule.csv":
            print(from_station, to_station)
            print("cost_function:", cost_function[0])
            result, total_distance, price, destination_arrival_time, days, train_departure_time = dijkstra(
                customGraph, from_station, to_station, schedule_file, cost_function[0])
            print("schedule file:", schedule_file)

        elif schedule_file == "schedule.csv":
            print(from_station, to_station)
            print("schedule file:", schedule_file)
            result, total_distance, price, destination_arrival_time, days, train_departure_time = dijkstra(
                customGraph_problem_csv, from_station, to_station, schedule_file, cost_function[0])
#    from_station = "BNVD"
#    to_station = "BLM"
#    #distance, stops
#    cost_function = "stops"
#    given_arrival_time = "17:25:00"

        if cost_function[0] == "distance":
            output = ""
            count = 0
            for train_no, start_node, end_node in result:
                if output!="":
                    temp_output = f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + " ; "+temp_output
                else:
                    temp_output= f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + temp_output

            result = output + f",{total_distance}"
            problem_no= problems[problem].ProblemNo+","
            write_to_file(result, problem_no, final_solution_file)
            print(result)
            # Cost function implemented for number of stops:
        elif cost_function[0] == "stops":
            output = ""
            stops_count = 0
            for train_no, start_node, end_node in result:
                if output != "":
                    temp_output = f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + " ; "+temp_output
                else:
                    temp_output= f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + temp_output
                stops_count += int(end_node)-int(start_node)


            result = output + f",{stops_count}"
            problem_no= problems[problem].ProblemNo+","
            write_to_file(result, problem_no, final_solution_file)
            print(result)
        elif cost_function[0] == "price":
            output = ""
            for train_no, start_node, end_node in result:
                if output!="":
                    temp_output =  f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + " ; "+temp_output
                else:
                    temp_output= f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + temp_output

            result = output + f",{price}"
            problem_no= problems[problem].ProblemNo+","
            write_to_file(result, problem_no, final_solution_file)
            print(result)
        elif cost_function[0] == "arrivaltime":
            given_arrival_time = cost_function[1]
            output = ""
            train_departure_time = train_departure_time.replace("'", "")

            given_arrival_time = given_arrival_time.replace("'", "")

            train_departure_time = datetime.strptime(train_departure_time, "%H:%M:%S")

            given_arrival_time = datetime.strptime(given_arrival_time, "%H:%M:%S")

            changeover_time = timedelta(minutes=10)

            new_departure_time = changeover_time + given_arrival_time

            if train_departure_time < new_departure_time:
                print("you will catch the train next day.")
                days += 1

            for train_no, start_node, end_node in result:
                if output!="":
                    temp_output =  f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + " ; "+temp_output
                else:
                    temp_output= f"{train_no} : {start_node} -> {end_node}"
                    temp_output = temp_output.replace("'", "")
                    output = output + temp_output
            dep_time = ""
            destination_arrival_time=str(destination_arrival_time)
            destination_arrival_time = destination_arrival_time.replace("'", "")
            if days != 0:
                if days < 10:
                    dep_time = "0" + str(days) + ":"+destination_arrival_time
                if days >= 10:
                    dep_time = str(days) + ":" + destination_arrival_time
            else:
                dep_time += destination_arrival_time

            result = output +f",{dep_time}"
            problem_no= problems[problem].ProblemNo+","
            write_to_file(result, problem_no, final_solution_file)
            print(result)

            #arrivaltime 17:25:00
