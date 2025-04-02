class Depot:
    def __init__(self, id, fleet_capacity: int, current_fleet: int, location: tuple):
        self.id = id
        self.fleet_capacity = fleet_capacity
        self.current_fleet = current_fleet
        self.location = location

    def pull_in(self):
        self.current_fleet += 1

    def pull_out(self):
        self.current_fleet -= 1