import { ApolloClient, ApolloProvider, InMemoryCache } from "@apollo/client";
import { CgMenuGridO } from "react-icons/cg";
import { MdOutlineDashboard, MdOutlineSettings } from "react-icons/md";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import "./App.css";

import Dashboard from "./components/Dashboard";
import Grid from "./components/Grid";
import Settings from "./components/Settings";

const client = new ApolloClient({
  uri: "http://localhost:8000/graphql",
  cache: new InMemoryCache(),
});

const App = () => {
  const location = useLocation();

  const pathname = location.pathname.split("/")[1];

  const pages = [
    { page: "dashboard", Icon: MdOutlineDashboard },
    { page: "grid", Icon: CgMenuGridO },
    { page: "settings", Icon: MdOutlineSettings },
  ];

  return (
    <ApolloProvider client={client}>
      <div className="w-full h-screen flex flex-col">
        <header className="sticky flex w-full top-0 start-0 h-14 items-center bg-teal-900 text-white text-sm p-1 font-medium">
          <nav>
            <ul className="flex flex-wrap text-center text-gray-500 m-4">
              {pages.map(({ page, Icon }) => {
                const c = page === pathname ? "text-white" : "";
                return (
                  <li key={page} className="me-2 m-1">
                    <Link
                      className={`flex items-center justify-center p-4 group ${c}`}
                      to={`/${page}`}
                    >
                      <Icon className="text-xl" />
                      {page.charAt(0).toUpperCase() + page.slice(1)}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        </header>
        <div className="grow">
          <Routes>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/grid" element={<Grid />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Routes>
        </div>
      </div>
    </ApolloProvider>
  );
};
export default App;
