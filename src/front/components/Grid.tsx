import { gql, useQuery } from "@apollo/client";
import { useKey, useMouse } from "@generalizers/react-utils";
import { useRef, useState } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  useNodesState,
} from "reactflow";

const Grid = () => {
  const position = useRef({ x: 0, y: 0 });
  const [gridName, setGridName] = useState("New grid");
  const [nodes, setNodes, _] = useNodesState([]);
  const [search, setSearch] = useState("");
  const { loading, error, data } = useQuery(gql`
    query {
      models {
        id
        name
      }
    }
  `);

  const [showList, setShowList] = useState(false);

  useKey("down", (e) => {
    if (e.code == "Space") {
      setShowList(true);
    }
  });

  const pos = position.current;

  useMouse("move", (e) => {
    pos.x = e.clientX;
    pos.y = e.clientY;
  });

  const addNode = (name: string) => {
    setShowList(false);
    setNodes((nds) =>
      nds.concat({ id: name, data: { label: name }, position: { x: 0, y: 0 } })
    );
    console.log(name);
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <p>Error : {error.message}</p>;

  const models = data.models;

  return (
    <div className="relative grow h-full flex">
      <ReactFlow
        className="absolute"
        nodes={nodes}
        onMoveStart={(_) => setShowList(false)}
      >
        <Background lineWidth={2} variant={BackgroundVariant.Cross} />
        {/* <Controls /> */}
      </ReactFlow>
      <div className="absolute w-full h-full text-white pointer-events-none [&>*]:pointer-events-auto">
        <button
          id="dropdownSearchButton"
          data-dropdown-toggle="dropdownSearch"
          data-dropdown-placement="bottom"
          className="w-40 m-2 bg-blue-700 hover:bg-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-left items-center"
          type="button"
        >
          {gridName}
        </button>
        <input
          value={gridName}
          className="p-2 bg-white border-2 outline-none text-black"
          onChange={(e) => setGridName(e.target.value)}
        />
        <button
          className="m-2 bg-blue-700 hover:bg-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center items-center "
          type="button"
          onClick={() => {}}
        >
          Save
        </button>
        <div
          id="dropdownSearch"
          className="hidden bg-white rounded-lg shadow w-60 dark:bg-gray-700"
        >
          <div className="p-3">
            <label htmlFor="input-group-search" className="sr-only">
              Search
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 rtl:inset-r-0 start-0 flex items-center ps-3 pointer-events-none">
                <svg
                  className="w-4 h-4 text-gray-500 dark:text-gray-400"
                  aria-hidden="true"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 20 20"
                >
                  <path
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"
                  />
                </svg>
              </div>
              <input
                type="text"
                id="input-group-search"
                className="block w-full p-2 ps-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                placeholder="Search user"
              />
            </div>
          </div>
          <ul
            className="h-48 px-3 pb-3 overflow-y-auto text-sm text-gray-700 dark:text-gray-200"
            aria-labelledby="dropdownSearchButton"
          >
            <li>
              <div className="flex items-center ps-2 rounded hover:bg-gray-100 dark:hover:bg-gray-600">
                <input
                  value=""
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-700 dark:focus:ring-offset-gray-700 focus:ring-2 dark:bg-gray-600 dark:border-gray-500"
                  onChange={() => {}}
                />
                <label
                  htmlFor="checkbox-item-11"
                  className="w-full py-2 ms-2 text-sm font-medium text-gray-900 rounded dark:text-gray-300"
                >
                  Bonnie Green
                </label>
              </div>
            </li>
          </ul>
        </div>
      </div>
      {showList && (
        <div
          className="absolute container w-60 h-80 bg-black text-white"
          style={{ left: pos.x, top: pos.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-between">
            <h1>List</h1>
            <button onClick={() => setShowList(false)}>X</button>
          </div>
          <div className="overflow-y-auto h-72 white">
            <ul>
              {models.map(({ name }: any) => {
                return (
                  <li key={name} onClick={(_) => addNode(name)}>
                    {name}
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default Grid;
