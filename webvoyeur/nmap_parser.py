import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PortInfo:
    """
    Represents information about a network port.

    This class is used to store and display details about a specific network port,
    including its number, protocol, state, associated service, and version. It provides
    an easy way to encapsulate port information and a string representation for display.
    """

    port: int
    protocol: str
    state: str
    service: str
    version: str

    def __str__(self):
        return (
            f"{self.port:>5}/{self.protocol:<3} {self.state:<8} {self.service:<12} {self.version}"
        )


class Host:
    """
    Represents a network host with its associated IP address and ports.

    This class is used to encapsulate information about a network host,
    including its IP address and a list of its open or associated ports.
    It provides utilities for representing the host as a string for
    human-readable display.
    """

    def __init__(self, ip: str, ports: list[PortInfo]):
        self.ip = ip
        self.ports = ports

    def __str__(self):
        host_str: str = f"Host: {self.ip}\n"
        for port in self.ports:
            host_str += f"\t{port}\n"
        return host_str


class NmapParser:
    """
    NmapParser is responsible for parsing Nmap XML output files and extracting
    information about hosts and their associated ports.

    This class provides a structured representation of the data included in an
    Nmap XML file. It extracts information about hosts, their IP addresses, and
    details about open ports such as protocol, state, associated service, and
    service version. This information is stored in an organized format to be accessed
    programmatically.

    Attributes:
        hosts (list[Host]): A list of Host objects containing parsed data about
        hosts and their associated ports.
    """

    def __init__(self, filepath: Path):
        self._tree = ET.parse(filepath)
        self._hosts = self._get_hosts()

    def _get_hosts(self):
        hosts: list[Host] = []
        for host in self._tree.findall("host"):
            ip = host.find("address").attrib["addr"]
            ports: list[PortInfo] = []
            for port in host.findall(".//port"):
                port_num = int(port.attrib["portid"])
                protocol = port.attrib["protocol"]
                state = port.find("state").attrib["state"]
                service = port.find("service").attrib["name"]
                version = self._get_version_str(port)
                ports.append(PortInfo(port_num, protocol, state, service, version))
            hosts.append(Host(ip, ports))
        return hosts

    def _get_version_str(self, port: ET.Element):
        product = ""

        try:
            product = port.find("service").attrib["product"]
        except KeyError:
            pass

        try:
            version = port.find("service").attrib["version"]
            if product == "":
                product = version
            else:
                product += f" {version}"

        except KeyError:
            pass

        try:
            extrainfo = port.find("service").attrib["extrainfo"]
            if product == "":
                product = f"{extrainfo}"
            else:
                product += f" ({extrainfo})"
        except KeyError:
            pass

        return product

    @property
    def hosts(self):
        return self._hosts

    def __str__(self):
        return str(self._hosts)


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is for demonstration purposes only. Please use the library in your own code."
    )
