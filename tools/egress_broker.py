#!/usr/bin/env python3
"""The only egress path out of the agent's network namespace.

tools/codex-articulated puts the sandbox in a namespace with no route to
anything except one veth peer, and no NAT behind it, so nothing inside can reach
the internet on its own. This process runs OUTSIDE that namespace, binds one
listener per allowed destination, and splices each to exactly one upstream.

The destination is fixed by the launcher at bind time. There is no CONNECT verb
and no Host header parsing, so a client inside the sandbox cannot name a
destination -- pointing a request at the API listener with
`Host: storage.googleapis.com` still lands on the API host. That is the property a
forward proxy does not have.

Every accepted connection is logged, so a run leaves an auditable trail of what it
talked to. It is not a record of what it tried: a destination that is not on the
list never gets here, because the kernel refuses it inside the namespace.
"""
import argparse
import os
import socket
import socketserver
import sys
import threading
import time


def make_logger(path):
    lock = threading.Lock()

    def log(msg):
        line = "%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
        with lock:
            # The launcher points stderr at the same file, so writing both would
            # double every line.
            if path:
                with open(path, "a") as fh:
                    fh.write(line)
            else:
                sys.stderr.write(line)
                sys.stderr.flush()

    return log


class Upstream:
    """One allowed destination, plus the addresses DNS last gave for it.

    The destination is a name, so every connection depends on the host's resolver
    still working. A name that resolved a moment ago has not moved, so remembering
    the answer turns a resolver hiccup into a retry instead of a lost run.

    Reusing a remembered address cannot redirect anyone: this process splices bytes
    and never terminates TLS, so the client inside the sandbox still handshakes
    against the name it asked for. An address that has genuinely gone stale fails
    that handshake rather than quietly passing.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.known = []

    def __str__(self):
        return "%s:%d" % (self.host, self.port)

    def lookup(self):
        infos = socket.getaddrinfo(
            self.host, self.port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
        self.known = [(info[0], info[4]) for info in infos]
        return self.known

    def connect(self, timeout=20):
        """Dial, preferring a fresh lookup and falling back to the last good one.

        Returns the socket and the reasons this connection was not made the normal
        way, which is empty on the normal path. Walking several addresses is not one
        of those reasons: a name with both an A and an AAAA record where only one is
        reachable is an ordinary Thursday, and reporting it would put a line in the
        egress log for every connection and bury the resolver failures worth seeing.

        Raises OSError naming both, so an UNREACHABLE line says what actually went
        wrong rather than only that something did.
        """
        degraded = []
        refused = []
        try:
            candidates = self.lookup()
        except OSError as exc:
            degraded.append("dns: %s" % exc)
            candidates = list(self.known)
            if candidates:
                degraded.append("used %d remembered address(es)" % len(candidates))
        for family, sockaddr in candidates:
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.settimeout(timeout)
                sock.connect(sockaddr)
                sock.settimeout(None)
                return sock, degraded
            except OSError as exc:
                sock.close()
                refused.append("%s: %s" % (sockaddr[0], exc))
        raise OSError(
            "; ".join(degraded + refused) or "no address for %s" % self
        )


def splice(src, dst):
    """Copy one direction until EOF, then break both halves out of recv()."""
    try:
        while True:
            buf = src.recv(65536)
            if not buf:
                break
            dst.sendall(buf)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class Relay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(target, log, label, attempts=3, backoff=0.5):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            peer = "%s:%d" % self.client_address[:2]
            dest = str(target)
            # Retrying here rather than leaving it to the client: the client is the
            # agent's model stream, and what it does with a failed connection is
            # give up on the turn.
            degraded = []
            failures = []
            upstream = None
            for attempt in range(attempts):
                try:
                    upstream, degraded = target.connect()
                    break
                except OSError as exc:
                    failures.append(str(exc))
                    if attempt + 1 < attempts:
                        time.sleep(backoff * (attempt + 1))
            if upstream is None:
                log("UNREACHABLE %s from=%s -> %s (%s)"
                    % (label, peer, dest, " | ".join(failures)))
                return
            if degraded or failures:
                # Connected, but not the way it was supposed to. Say so: this is the
                # only place a resolver going soft is visible before it costs a run.
                log("DEGRADED %s from=%s -> %s (%s)"
                    % (label, peer, dest, " | ".join(failures + degraded)))
            log("ALLOW %s from=%s -> %s" % (label, peer, dest))
            pump = threading.Thread(
                target=splice, args=(self.request, upstream), daemon=True
            )
            pump.start()
            splice(upstream, self.request)
            pump.join()

    return Handler


def qname_of(query):
    """The question name from a DNS query, dotted. Empty if it does not parse."""
    labels = []
    offset = 12  # fixed header
    try:
        while True:
            length = query[offset]
            if length == 0 or length & 0xC0:  # end of name, or a pointer
                break
            offset += 1
            # Wire format is ASCII, punycode included. Decoding as ASCII keeps an
            # internationalised name in its on-the-wire form, which is the form
            # worth having in an audit log anyway. Not idna: that codec rejects
            # every error handler, so one odd label would lose the whole name.
            labels.append(query[offset:offset + length].decode("ascii", "replace"))
            offset += length
    except IndexError:
        return ""
    # The name is attacker-chosen and this log is the evidence, so keep newlines
    # and escape sequences out of it: nobody should be able to forge a line.
    return "".join(c if c.isprintable() and c != " " else "?"
                   for c in ".".join(labels))


class Sinkhole(socketserver.ThreadingUDPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_dns_handler(log):
    """Answer every query NXDOMAIN, and write down what was asked for.

    The allowed destinations are in the sandbox's hosts file, and glibc reads that
    before it ever asks a nameserver, so anything arriving here is by definition a
    name the run was not given. Two things then become visible that were not:
    an agent going looking for data it should be reconstructing, and the model
    endpoint being renamed under a live run -- which without this looks like an
    unexplained stall rather than one line naming the host that moved.
    """

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            query, sock = self.request
            if len(query) < 12:
                return
            name = qname_of(query) or "<unparsed>"
            log("DNS-REFUSED %s from=%s" % (name, self.client_address[0]))
            header = query[:2] + b"\x81\x83" + query[4:6] + b"\x00\x00\x00\x00\x00\x00"
            sock.sendto(header + query[12:], self.client_address)

    return Handler


def parse_map(spec):
    listen, target = spec.split("=", 1)
    lhost, lport = listen.rsplit(":", 1)
    thost, tport = target.rsplit(":", 1)
    return (lhost, int(lport)), Upstream(thost, int(tport))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="LISTEN_IP:PORT=UPSTREAM_HOST:PORT",
        help="one allowed destination; repeat per destination",
    )
    ap.add_argument(
        "--dns-sinkhole",
        metavar="IP:PORT",
        help="answer DNS here with NXDOMAIN and log the name asked for; run this "
        "instance inside the namespace, where the sandbox's resolv.conf points",
    )
    ap.add_argument("--log", help="append the egress trail here as well as stderr")
    ap.add_argument("--label", default="egress", help="tag for log lines")
    ap.add_argument(
        "--ready-file",
        help="created once every listener is bound, so the launcher can wait "
        "instead of sleeping",
    )
    ap.add_argument(
        "--verify-upstreams",
        action="store_true",
        help="dial each upstream once before reporting ready, and refuse to "
        "start if one does not answer",
    )
    args = ap.parse_args()

    if not args.map and not args.dns_sinkhole:
        ap.error("nothing to do: pass --map and/or --dns-sinkhole")

    log = make_logger(args.log)
    servers = []
    if args.dns_sinkhole:
        sink_ip, sink_port = args.dns_sinkhole.rsplit(":", 1)
        servers.append(Sinkhole((sink_ip, int(sink_port)), make_dns_handler(log)))
        log("LISTEN %s dns-sinkhole %s" % (args.label, args.dns_sinkhole))
    # One Upstream per destination, shared with its listener, so the verification
    # below also primes the address it remembers.
    upstreams = []
    for spec in args.map:
        listen, target = parse_map(spec)
        upstreams.append(target)
        label = "%s[%s]" % (args.label, target)
        # Bind before backgrounding: a bind failure must fail the launcher, not
        # leave the sandbox running with a hole where a listener should be.
        server = Relay(listen, make_handler(target, log, label))
        servers.append(server)
        log("LISTEN %s %s:%d -> %s" % (args.label, listen[0], listen[1], target))

    if args.verify_upstreams:
        # A listener binds whether or not the far end exists, so dialing the relay
        # proves nothing about the upstream -- the accept happens first and the
        # upstream connect only follows. Check the far end here, where it is, and
        # refuse to report ready if it does not answer: never reported ready means
        # the launcher fails at startup instead of the agent losing its model
        # partway through a run.
        for target in upstreams:
            try:
                sock, problems = target.connect()
            except OSError as exc:
                log("UPSTREAM DEAD %s (%s)" % (target, exc))
                return 1
            sock.close()
            if problems:
                log("UPSTREAM DEGRADED %s (%s)" % (target, "; ".join(problems)))
            log("UPSTREAM OK %s" % target)

    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()

    if args.ready_file:
        with open(args.ready_file, "w") as fh:
            fh.write("%d\n" % os.getpid())

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
